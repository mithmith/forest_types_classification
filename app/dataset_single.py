import json
import os
import random
import re
import shutil
import warnings
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from loguru import logger
from rasterio.features import Affine
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy.ndimage import rotate
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform
from torchsummary import summary
from torchview import draw_graph
from torchviz import make_dot
from tqdm import tqdm
from xgboost import XGBClassifier

import app.utils.geo_mask as geo_mask
import app.utils.veg_index as veg_index
from app.services.base import BoundingBox

# from app.train import evaluate, load_model

warnings.filterwarnings("ignore", category=rasterio.features.ShapeSkipWarning)


class ForestTypesDataset:
    def __init__(
        self,
        geojson_masks_dir: Path | None = None,
        sentinel_root_path: Path | None = None,
        dataset_path: Path | None = None,
        forest_model_path: Path | None = None,
        crop_bboxes_dir: Path | None = None,
    ) -> None:
        self.image_shape = (512, 512)
        self.bands_regex_list = {
            "red": r"(\w*)_(\d{8}T\d*)_(B04|B04_10m)\.jp2",
            "green": r"(\w*)_(\d{8}T\d*)_(B03|B03_10m)\.jp2",
            "blue": r"(\w*)_(\d{8}T\d*)_(B02|B02_10m)\.jp2",
            "nir": r"(\w*)_(\d{8}T\d*)_(B08|B08_10m)\.jp2",
            "swir1": r"(\w*)_(\d{8}T\d*)_(B11|B11_10m)\.jp2",
            "swir2": r"(\w*)_(\d{8}T\d*)_(B12|B12_10m)\.jp2",
        }
        self.sentinel_root = sentinel_root_path
        self.generated_dataset_path = dataset_path
        if geojson_masks_dir is not None and geojson_masks_dir.exists():
            self.geojson_files = list(geojson_masks_dir.glob("*.geojson"))
        if sentinel_root_path is not None and sentinel_root_path.exists():
            self.images_files = [f for f in os.listdir(sentinel_root_path)]
        self.dataset_length = len(list(self.generated_dataset_path.glob("*_mask.tif")))
        self.forest_model_path = forest_model_path
        self.crop_bboxes_dir = crop_bboxes_dir

    def __len__(self):
        return self.dataset_length

    def find_geojson_path_for_img(self, img: str):
        first_part = img.split("_")[2]
        second_part = img.split("_")[5]
        for s in self.geojson_files:
            if first_part in str(s) and second_part in str(s):
                mask_path = Path(s)
                return mask_path
        return None

    def get_info_from_img(self, img_name: str):
        first_part = img_name.split("_")[2]
        second_part = img_name.split("_")[5]

        with rasterio.open(self.sentinel_root / Path(img_name) / f"{second_part}_{first_part}_B04_10m.jp2") as src:
            img_width, img_height = src.width, src.height
            transform_matrix = src.transform
            crs = src.crs

        return transform_matrix, img_width, img_height, crs

    @staticmethod
    def save_cropped_region(tif_path: Path, box: BoundingBox, output_path: Path):
        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(tif_path) as src:
                # Создаем окно для вырезки области по координатам box
                window = Window(box.minx, box.miny, box.maxx - box.minx, box.maxy - box.miny)
                transform = src.window_transform(window)

                # Читаем данные окна
                cropped_data: np.ndarray = src.read(1, window=window)

                # Сохраняем вырезанные данные в новый tif файл
                profile: dict = src.profile.copy()
                profile.update(
                    {
                        "height": box.maxy - box.miny,
                        "width": box.maxx - box.minx,
                        "transform": transform,
                        "driver": "GTiff",
                        "compress": "lzw",
                        "tiled": True,
                    }
                )

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(cropped_data, 1)

    @staticmethod
    def save_mask(
        mask: np.ndarray,
        img_height: int,
        img_width: int,
        transform_matrix: Affine,
        target_crs: str,
        output_tif_path: Path,
    ):
        # Сохранение маски в GeoTIFF
        mask_profile = {
            "driver": "GTiff",
            "height": img_height,
            "width": img_width,
            "count": 1,
            "dtype": mask.dtype,
            "crs": target_crs,
            "transform": transform_matrix,
        }

        with rasterio.open(output_tif_path, "w", **mask_profile) as dst:
            dst.write(mask, 1)

    @staticmethod
    def get_random_bbox(width: int, height: int, bbox_shape: tuple[int, int]) -> BoundingBox:
        """
        Генерирует случайный bounding box внутри изображения с заданной шириной и высотой.

        Args:
            width (int): ширина изображения.
            height (int): высота изображения.
            bbox_shape (Tuple[int, int]): ширина и высота bounding box (в пикселях).

        Returns:
            BoundingBox: случайно сгенерированный bounding box внутри изображения.
        """
        bbox_width, bbox_height = bbox_shape
        max_x = max(0, width - bbox_width)
        max_y = max(0, height - bbox_height)
        minx = random.randint(0, max_x) if max_x > 0 else 0
        miny = random.randint(0, max_y) if max_y > 0 else 0
        maxx = minx + bbox_width
        maxy = miny + bbox_height
        return BoundingBox(minx, maxx, miny, maxy, 0, 0)

    @staticmethod
    def get_bbox_from_geojson(geojson_path: Path, target_crs: pyproj.CRS) -> BoundingBox:
        """
        Reads a GeoJSON file, reprojects the geometry from its source CRS
        (assumed to be EPSG:4326 if not specified) to the target_crs,
        and extracts the bounding box as a BoundingBox instance.
        """
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)

        first_feature = geojson_data["features"][0]
        geom = shape(first_feature["geometry"])

        source_crs = pyproj.CRS("EPSG:4326")

        if source_crs != target_crs:
            transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
            geom = transform(transformer.transform, geom)

        minx, miny, maxx, maxy = geom.bounds
        return BoundingBox(minx, maxx, miny, maxy, 0, 0)

    @staticmethod
    def warp_image_to_temp(src_image_path: Path, crop_bbox: BoundingBox, temp_folder: Path) -> Path:
        """
        Crops the source image using crop_bbox (in the image's pixel coordinate space)
        and saves it to a temporary folder.
        """
        temp_folder.mkdir(exist_ok=True)
        temp_output = temp_folder / f"{src_image_path.stem}_cropped.tif"

        with rasterio.Env(GDAL_PAM_ENABLED="NO"):
            with rasterio.open(src_image_path) as src:
                # Convert crop_bbox from map coordinates to pixel indices.
                row_min, col_min = src.index(crop_bbox.minx, crop_bbox.maxy)
                row_max, col_max = src.index(crop_bbox.maxx, crop_bbox.miny)

                if row_min > row_max:
                    row_min, row_max = row_max, row_min
                if col_min > col_max:
                    col_min, col_max = col_max, col_min

                window_height = row_max - row_min
                window_width = col_max - col_min

                if window_width <= 0 or window_height <= 0:
                    raise ValueError(
                        f"Invalid crop window dimensions: width={window_width}, height={window_height}. "
                        "Ensure that the GeoJSON crop box overlaps the image area and is reprojected to the image's CRS."
                    )

                window = Window(col_off=col_min, row_off=row_min, width=window_width, height=window_height)
                transform = src.window_transform(window)

                cropped_data = src.read(1, window=window)
                profile = src.profile.copy()
                profile.update(
                    {
                        "height": window_height,
                        "width": window_width,
                        "transform": transform,
                        "driver": "GTiff",
                        "compress": "lzw",
                        "tiled": True,
                    }
                )

                with rasterio.open(temp_output, "w", **profile) as dst:
                    dst.write(cropped_data, 1)

        return temp_output

    def generate_dataset(self, temp_folder: Path):
        if self.generated_dataset_path is None:
            raise ValueError("Generate dataset path is required")
        if not self.geojson_files:
            raise ValueError("GeoJSON files are required")

        num_images = len(self.images_files)
        if num_images == 0:
            raise ValueError("No images found to generate samples from.")

        total_samples = 0

        with tqdm(total=num_images, desc="Images processed", unit="image") as pbar:
            for img_file in self.images_files:
                full_img_path = self.sentinel_root / img_file

                geojson_mask_path = self.find_geojson_path_for_img(img_file)
                if geojson_mask_path is None:
                    logger.info(f"Mask not found for image {img_file}, skipping.")
                    pbar.update(1)
                    continue

                if self.crop_bboxes_dir is not None:
                    region_bbox = img_file.split("_")[-2]
                    image_crop_bbox_path = self.crop_bboxes_dir.joinpath("crop_bbox_" + region_bbox + ".geojson")
                    if image_crop_bbox_path.exists():
                        with rasterio.open(
                            (full_img_path / f"{region_bbox}_{img_file.split('_')[2]}_B04_10m.jp2")
                        ) as src:
                            target_crs = src.crs
                        crop_bbox = self.get_bbox_from_geojson(image_crop_bbox_path, target_crs)
                        cropped_img_path = self.warp_image_to_temp(
                            (full_img_path / f"{region_bbox}_{img_file.split('_')[2]}_B04_10m.jp2"),
                            crop_bbox,
                            temp_folder,
                        )
                        using_temp = True
                else:
                    cropped_img_path = full_img_path
                    using_temp = False

                with rasterio.open(cropped_img_path) as src:
                    cropped_width, cropped_height = src.width, src.height
                    transform_matrix = src.transform
                    crs = src.crs

                with open(geojson_mask_path) as f:
                    geojson_dict = json.load(f)

                # Create the mask for the cropped image.
                mask = geo_mask.mask_from_geojson(geojson_dict, (cropped_height, cropped_width), transform_matrix, crs)
                if mask is None:
                    logger.info(f"Mask is empty for image {img_file}, skipping.")
                    if using_temp and temp_folder.exists():
                        shutil.rmtree(temp_folder)
                    pbar.update(1)
                    continue

                samples_per_slice = (cropped_width // 512) * (cropped_height // 512) * 4
                logger.info(f"For image {img_file}, computed samples_per_slice: {samples_per_slice}")

                generated_samples = 0
                not_found = 0

                with tqdm(
                    total=samples_per_slice, desc="Generating Samples", unit="sample", position=0, leave=True
                ) as sub_pbar:
                    while generated_samples < samples_per_slice:
                        box = self.get_random_bbox(cropped_width, cropped_height, self.image_shape)
                        mask_region = mask[box.miny : box.maxy, box.minx : box.maxx]
                        if np.count_nonzero(mask_region) <= 1000:
                            not_found += 1
                            if not_found > 500:
                                logger.info(
                                    "Not found enough valid regions after 500 attempts, moving to next image..."
                                )
                                if using_temp and temp_folder.exists():
                                    shutil.rmtree(temp_folder)
                                break
                            continue

                        rnd_num = random.randint(100000, 999999)
                        crop_mask_path = self.generated_dataset_path / f"{rnd_num}_mask.tif"
                        cropped_transform_matrix = window_transform(
                            Window(box.minx, box.miny, box.width, box.height), transform_matrix
                        )
                        self.save_mask(
                            mask_region, box.height, box.width, cropped_transform_matrix, crs, crop_mask_path
                        )

                        # Process each band from the cropped image.
                        for band_key, band_regex in self.bands_regex_list.items():
                            band_path = self.get_band_images(self.sentinel_root / img_file, band_regex)
                            if band_path.exists():
                                if self.crop_bboxes_dir is not None and image_crop_bbox_path.exists():
                                    if not (temp_folder / f"{band_path.stem}_cropped.tif").exists():
                                        cropped_band_path = self.warp_image_to_temp(band_path, crop_bbox, temp_folder)
                                    else:
                                        cropped_band_path = temp_folder / f"{band_path.stem}_cropped.tif"
                                    crop_output_path = self.generated_dataset_path / f"{rnd_num}_{band_key}.tif"
                                    self.save_cropped_region(cropped_band_path, box, crop_output_path)
                                else:
                                    crop_output_path = self.generated_dataset_path / f"{rnd_num}_{band_key}.tif"
                                    self.save_cropped_region(band_path, box, crop_output_path)
                            else:
                                logger.warning(f"Band file not found for {band_key} in image {img_file}")

                        generated_samples += 1
                        total_samples += 1
                        sub_pbar.update(1)

                logger.info(f"Total samples generated from image {img_file}: {generated_samples}")

                if using_temp and temp_folder.exists():
                    shutil.rmtree(temp_folder)
                pbar.update(1)

            logger.info(f"Total samples generated: {total_samples}")

    @staticmethod
    def get_band_images(img_path: Path, band_regex: str) -> Path:
        """
        Находит и возвращает пути к файлам заданного бэнда в папках img_path.
        """
        pattern = re.compile(band_regex)

        def find_band_file(folder_path: Path) -> Path:
            for file in folder_path.iterdir():
                if pattern.match(file.name):
                    return file
            raise FileNotFoundError(f"Файл для бэнда не найден в папке: {folder_path}")

        tif_path = find_band_file(img_path)
        return tif_path

    @staticmethod
    def add_salt_and_pepper_noise(image: np.ndarray, salt_percent: float, pepper_percent: float) -> np.ndarray:
        """
        Добавляет шум типа "соль и перец" к изображению.

        :param image: Входной массив изображения (H, W).
        :param salt_percent: Доля пикселей, заменяемых на 1 (соль).
        :param pepper_percent: Доля пикселей, заменяемых на 0 (перец).
        :return: Изображение с добавленным шумом.
        """
        noisy_image = image.copy()
        total_pixels = image.size

        # Случайные индексы для соли
        num_salt = int(total_pixels * salt_percent)
        salt_coords = (np.random.randint(0, image.shape[0], num_salt), np.random.randint(0, image.shape[1], num_salt))

        # Случайные индексы для перца
        num_pepper = int(total_pixels * pepper_percent)
        pepper_coords = (
            np.random.randint(0, image.shape[0], num_pepper),
            np.random.randint(0, image.shape[1], num_pepper),
        )

        # Добавляем шум
        noisy_image[salt_coords] = 1  # Соль
        noisy_image[pepper_coords] = 0  # Перец

        return noisy_image

    @staticmethod
    def add_gaussian_noise(image: np.ndarray, std_dev: float = 0.01) -> np.ndarray:
        noise = np.random.normal(0, std_dev, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)

    @staticmethod
    def add_random_rotation_and_flip(
        bands_1: list[np.ndarray], bands_2: list[np.ndarray], mask: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """
        Применяет одинаковые случайные повороты (90°, 180°, 270°) и отражения
        (по вертикали и/или горизонтали) к двум наборам данных и маске.
        """
        # Случайный поворот
        angle = np.random.choice([0, 90, 180, 270])
        # logger.debug(f"Angle of rotation: {angle}")
        # logger.debug(f"bands_1.shape: {len(bands_1)}, {bands_1[0].shape}")
        # logger.debug(f"bands_2.shape: {len(bands_2)}, {bands_2[0].shape}")
        # logger.debug(f"mask.shape: {mask.shape}")
        if angle != 0:
            bands_1 = [rotate(band, angle, axes=(0, 1), reshape=False) for band in bands_1]
            bands_2 = [rotate(band, angle, axes=(0, 1), reshape=False) for band in bands_2]
            mask = rotate(mask, angle, axes=(0, 1), reshape=False)

        # Случайное отражение
        if np.random.rand() > 0.5:  # Отражение по ширине
            bands_1 = [np.flip(band, axis=1) for band in bands_1]
            bands_2 = [np.flip(band, axis=1) for band in bands_2]
            mask = np.flip(mask, axis=1)

        if np.random.rand() > 0.5:  # Отражение по высоте
            bands_1 = [np.flip(band, axis=0) for band in bands_1]
            bands_2 = [np.flip(band, axis=0) for band in bands_2]
            mask = np.flip(mask, axis=0)

        return bands_1, bands_2, mask

    @staticmethod
    def prepare_forest_data(input_data: np.ndarray) -> np.ndarray:
        """Подготавливает данные для модели."""
        # Теперь форма: (количество_изображений, высота, ширина, количество_каналов)
        # logger.debug(f"input_data shape: {input_data.shape}")
        # images = np.squeeze(input_data, axis=1)

        # Если форма входного массива (высота, ширина, количество_каналов), ничего не переставляем
        if input_data.ndim == 3:
            # Сглаживаем первые две размерности (высота и ширина) и оставляем каналы
            prepared_data = input_data.reshape(-1, input_data.shape[2])
        else:
            # Переставляем порядок размерностей для удобства
            # Из (количество_изображений, высота, ширина, количество_каналов) делаем (высота, ширина, количество_изображений, количество_каналов)
            images = np.transpose(input_data, (1, 2, 0, 3))

            # Сглаживаем первые две размерности (высота и ширина) и объединяем последние две (количество_изображений и количество_каналов)
            # Форма: (количество_пикселей, количество_изображений * количество_каналов)
            prepared_data = images.reshape(images.shape[0] * images.shape[1], -1)

        # logger.debug(f"prepared_data shape: {prepared_data.shape}")
        return prepared_data

    @staticmethod
    def create_forest_mask(sample_id: str, dataset_path: Path, forest_model_path: Path) -> np.ndarray:
        bands_list = ["blue", "nir", "swir1", "swir2", "ndvi", "ndmi", "evi", "ndwi", "nbr", "mndwi"]

        bands_path_list = {}

        for band_key in ["red", "green", "blue", "nir", "swir1", "swir2"]:
            bands_path_list[band_key] = dataset_path / f"{sample_id}_{band_key}.tif"

        stacked_array, _ = veg_index.calculate_image_data(bands_path_list, bands_list, level=1)

        forest_model = XGBClassifier()
        forest_model.load_model(forest_model_path)
        prepared_data = ForestTypesDataset.prepare_forest_data(stacked_array)
        predictions = forest_model.predict(prepared_data)

        num_classes = predictions.shape[1]
        prediction = predictions.reshape(stacked_array.shape[0], stacked_array.shape[1], num_classes)
        return prediction[:, :, 0]

    def get_next_generated_sample(
        self, verbose: bool = False, exclude_nir=True, exclude_fMASK=True
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        mask_files = list(self.generated_dataset_path.glob("*_mask.tif"))
        self.dataset_length = len(mask_files)
        random.shuffle(mask_files)

        for mask_file in mask_files:
            # Extract sample identifier from the mask filename
            sample_id = mask_file.stem.split("_")[0]

            # Load the mask
            with rasterio.open(mask_file) as mask_src:
                mask = mask_src.read(1)

            # Load and stack all bands for the sample
            band_data = []
            for band_key in self.bands_regex_list.keys():
                if "nir" in str(band_key) and exclude_nir:
                    continue
                if "swir1" in str(band_key) or "swir2" in str(band_key):
                    continue
                band_file = self.generated_dataset_path / f"{sample_id}_{band_key}.tif"
                if band_file.exists():
                    with rasterio.open(band_file) as band_src:
                        band = veg_index.preprocess_band(band_src.read(1))
                        band = self.add_salt_and_pepper_noise(band, salt_percent=0.01, pepper_percent=0.01)
                        band = self.add_gaussian_noise(band)
                        band_data.append(band)
                else:
                    logger.warning(f"Missing band file: {band_file}, skipping this sample.")
                    continue

            if not exclude_fMASK:
                band_data.append(
                    self.create_forest_mask(sample_id, self.generated_dataset_path, self.forest_model_path)
                )

            if not band_data:
                logger.warning(f"No valid bands found for sample {sample_id}, skipping.")
                continue

            # Stack bands into a 3D array
            bands_stacked = np.stack(band_data, axis=0)

            # Visualize the data if verbose is True
            if verbose:
                plt.figure(figsize=(12, 6))
                for i, band in enumerate(bands_stacked, 1):
                    plt.subplot(1, len(bands_stacked) + 1, i)
                    plt.imshow(band, cmap="gray")
                    plt.title(f"Band {i}")

                plt.subplot(1, len(bands_stacked) + 1, len(bands_stacked) + 1)
                plt.imshow(mask, cmap="gray")
                plt.title("Mask")
                plt.show()

            # Yield the stacked bands and corresponding mask
            yield bands_stacked, mask

    def model_summary_structure(self, model, name="model", exclude_nir=False, exclude_fMASK=False):
        sample, mask = next(self.get_next_generated_sample(exclude_nir=exclude_nir, exclude_fMASK=exclude_fMASK))
        summary(model, sample.shape, batch_size=1, device="cpu")
        output = model(model.prepare_input(sample))
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.format = "png"
        dot.render(name)
        model_graph = draw_graph(model, input_size=sample.shape, expand_nested=True)
        model_graph.visual_graph
