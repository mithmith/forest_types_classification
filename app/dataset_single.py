import json
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from osgeo import gdal
from loguru import logger
from rasterio.features import Affine
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import Polygon, MultiPolygon
from scipy.ndimage import rotate
from torchsummary import summary
from torchview import draw_graph
from torchviz import make_dot
from tqdm import tqdm
from xgboost import XGBClassifier

import app.utils.geo_mask as geo_mask
import app.utils.veg_index as veg_index
from app.services.base import BoundingBox
from app.train import evaluate, load_model

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
        self.dataset_length = 0
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

    def normalize_geojson_coordinates(self, geojson_data, img_width, img_height):
        """
        Normalize GeoJSON coordinates and scale them to image dimensions.

        Args:
            geojson_data (dict): The GeoJSON data.
            img_width (int): The width of the image.
            img_height (int): The height of the image.

        Returns:
            normalized_polygon (shapely.geometry.Polygon): The polygon scaled to image dimensions.
        """
        from shapely.geometry import shape

        # Extract the bounds of the GeoJSON data
        feature = geojson_data["features"][0]
        geometry = shape(feature["geometry"])
        geo_bounds = geometry.bounds  # (minx, miny, maxx, maxy)

        # Define min and max longitude/latitude (GeoJSON CRS84)
        min_lon, min_lat, max_lon, max_lat = geo_bounds

        # Define the normalization function
        def normalize(coord, min_val, max_val, scale):
            return (coord - min_val) / (max_val - min_val) * scale

        def normalize_polygon(polygon):
            normalized_coords = []
            for lon, lat in polygon.exterior.coords:
                x = normalize(lon, min_lon, max_lon, img_width)  # Normalize longitude
                y = normalize(lat, min_lat, max_lat, img_height)  # Normalize latitude
                normalized_coords.append((x, y))
            return Polygon(normalized_coords)

        if isinstance(geometry, MultiPolygon):
            normalized_polygons = [normalize_polygon(poly) for poly in geometry.geoms]
            normalized_geometry = MultiPolygon(normalized_polygons)
        elif isinstance(geometry, Polygon):
            normalized_geometry = normalize_polygon(geometry)

        return normalized_geometry

    def get_random_bbox_within_crop_bbox(self, width: int, height: int, bbox_shape: tuple[int, int],
                                         normalized_polygon) -> BoundingBox:
        """
        Generate a random bounding box constrained by a GeoJSON bounding area.

        Args:
            width (int): Width of the image.
            height (int): Height of the image.
            bbox_shape (tuple[int, int]): Width and height of the bounding box.
        Returns:
            BoundingBox: Randomly generated bounding box within the constrained area.
        """

        # Extract the geometry and get bounds
        minx, miny, maxx, maxy = normalized_polygon.bounds

        # Convert GeoJSON bounds to image coordinates
        bbox_width, bbox_height = bbox_shape

        # Ensure the random bounding box fits inside the normalized polygon
        max_x = max(0, int(maxx) - bbox_width)
        max_y = max(0, int(maxy) - bbox_height)

        # Generate random coordinates within the constraints
        rand_x = random.randint(int(minx), max_x) if max_x > minx else int(minx)
        rand_y = random.randint(int(miny), max_y) if max_y > miny else int(miny)

        # Calculate the bottom-right corner
        maxx = rand_x + bbox_width
        maxy = rand_y + bbox_height

        return BoundingBox(rand_x, maxx, rand_y, maxy, 0, 0)

    def generate_dataset(self, target_samples: int = 100):
        if self.generated_dataset_path is None:
            raise ValueError("Generate dataset path is required")
        if not self.geojson_files:
            raise ValueError("GeoJSON files are required")

        num_images = len(self.images_files)
        if num_images == 0:
            raise ValueError("No images found to generate samples from.")

        samples_per_image = target_samples // num_images

        total_samples = 0

        with tqdm(total=target_samples, desc="Generating Samples", unit="sample") as pbar:
            while total_samples < target_samples:
                for img_file in self.images_files:
                    geojson_mask_path = self.find_geojson_path_for_img(img_file)
                    if geojson_mask_path is None:
                        logger.info(f"Mask not found for image {img_file}, skipping.")
                        continue

                    transform_matrix, img_width, img_height, crs = self.get_info_from_img(img_file)
                    with open(geojson_mask_path) as f:
                        geojson_dict = json.load(f)

                    mask = geo_mask.mask_from_geojson(geojson_dict, (img_height, img_width), transform_matrix, crs)

                    generated_samples = 0
                    not_found = 0
                    region_bbox = img_file.split("_")[-2]
                    if self.crop_bboxes_dir is not None:
                        image_crop_bbox_path = self.crop_bboxes_dir.joinpath("crop_bbox_" + region_bbox + '.geojson')
                        if image_crop_bbox_path.exists():
                            with open(image_crop_bbox_path, 'r') as f:
                                geojson_data = json.load(f)
                                normalized_polygon = self.normalize_geojson_coordinates(geojson_data, img_width, img_height)

                    while generated_samples < samples_per_image:
                        if image_crop_bbox_path is not None and image_crop_bbox_path.exists():
                            box = self.get_random_bbox_within_crop_bbox(img_width, img_height, self.image_shape, normalized_polygon)
                        else:
                            box = self.get_random_bbox(img_width, img_height, self.image_shape)
                        mask_region = mask[box.miny : box.maxy, box.minx : box.maxx]
                        if np.count_nonzero(mask_region) <= 1000:
                            not_found += 1
                            if not_found > 500:
                                logger.info("Not found anything after 500 attempts, skipping to next image...")
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

                        for band_key, band_regex in self.bands_regex_list.items():
                            band_path = self.get_band_images(self.sentinel_root / img_file, band_regex)
                            if band_path.exists():
                                crop_output_path = self.generated_dataset_path / f"{rnd_num}_{band_key}.tif"
                                self.save_cropped_region(band_path, box, crop_output_path)

                        generated_samples += 1
                        total_samples += 1
                        pbar.update(1)

                        if total_samples >= target_samples:
                            break

        logger.info(f"Total samples generated: {total_samples}")

    @staticmethod
    def get_band_images(img_path: Path, band_regex: str) -> Path:
        """
        Находит и возвращает пути к файлам заданного бэнда в папках img_path.

        Args:
            img_path (Path): Путь к первой папке со снимком.
            band_regex (str): Регулярное выражение для поиска файлов бэнда.

        Returns:
            Tuple[Path, Path]: Пути к файлам бэнда в img1_path и img2_path.
        """
        pattern = re.compile(band_regex)

        def find_band_file(folder_path: Path) -> Path:
            for file in folder_path.iterdir():
                if pattern.match(file.name):
                    return file
            raise FileNotFoundError(f"Файл для бэнда не найден в папке: {folder_path}")

        # Найти файлы бэнда в обеих папках
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

    def prepare_forest_data(self, input_data: np.ndarray) -> np.ndarray:
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

    def create_forest_mask(self, sample_id: str) -> np.ndarray:

        bands_list = ["blue", "nir", "swir1", "swir2", "ndvi", "ndmi", "evi", "ndwi", "nbr", "mndwi"]

        bands_path_list = {}

        for band_key in ["red", "green", "blue", "nir", "swir1", "swir2"]:
            bands_path_list[band_key] = self.generated_dataset_path / f"{sample_id}_{band_key}.tif"

        stacked_array, _ = veg_index.calculate_image_data(bands_path_list, bands_list, level=1)

        forest_model = XGBClassifier()
        forest_model.load_model(self.forest_model_path)

        prepared_data = self.prepare_forest_data(stacked_array)
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
                band_data.append(self.create_forest_mask(sample_id))

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

    def predict_sample_from_dataset(
        self, model, model_path, sample_num: str, exclude_nir=False, exclude_fMASK=False, visualise=False
    ):
        features_names = ["red", "green", "blue"]

        if not exclude_nir:
            features_names.append("nir")

        input_tensor = []
        output_img = []
        for feature_name in features_names:
            with rasterio.open(self.generated_dataset_path / f"{sample_num}_{feature_name}.tif") as f:
                if feature_name != "nir":
                    in_ds = gdal.OpenEx(self.generated_dataset_path / f"{sample_num}_{feature_name}.tif")
                    out_ds = gdal.Translate('/vsimem/in_memory_output.tif', in_ds)
                    out_arr = out_ds.ReadAsArray()
                    output_img.append(out_arr)
                input_tensor.append(veg_index.preprocess_band(f.read(1)))

        ground_truth_tensor = []
        with rasterio.open(self.generated_dataset_path / f"{sample_num}_mask.tif") as f:
            ground_truth_tensor.append(f.read(1))

        if not exclude_fMASK:
            input_tensor.append(self.create_forest_mask(sample_num))

        input_tensor = np.array(input_tensor)
        loaded_model = load_model(model, model_path)

        predict_mask = evaluate(loaded_model, input_tensor)

        if visualise:
            output_img = np.transpose(output_img, (1, 2, 0))
            normalized_rgb = np.zeros_like(output_img, dtype=np.float32)  # Создаём пустой массив для нормализации
            for channel in range(output_img.shape[2]):  # По каждому каналу (R, G, B)
                channel_data = output_img[:, :, channel]
                normalized_rgb[:, :, channel] = (channel_data - channel_data.min()) / (
                        channel_data.max() - channel_data.min() + 1e-6
                )  # Добавляем 1e-6 для избежания деления на 0
            # Increase brightness by scaling up values (factor 1.5 can be adjusted)
            brightness_factor = 3
            brightened_rgb = np.clip(normalized_rgb * brightness_factor, 0, 1)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(brightened_rgb)
            plt.imshow(predict_mask.clip(0.3, 0.75), cmap="hot", alpha=0.5)
            plt.title("RGB Image + Model Mask")
            plt.subplot(1, 3, 2)
            plt.imshow(brightened_rgb)
            plt.imshow(np.squeeze(ground_truth_tensor, axis=0).clip(0.3, 0.75), cmap="hot", alpha=0.5)
            plt.title("RGB Image + Ground Truth Mask")
            plt.subplot(1, 3, 3)
            ground_truth = np.squeeze(ground_truth_tensor, axis=0).clip(0.3, 0.75)
            plt.imshow(ground_truth, cmap="grey")
            plt.imshow(predict_mask.clip(0.3, 0.75), cmap="hot", alpha=0.5)
            plt.title("Ground Truth Mask + Model Mask")
            plt.tight_layout()
            plt.show()

        return predict_mask

    def inference_test(self, model, model_path, sample_num: str, num_runs, exclude_nir=False, exclude_fMASK=False):
        features_names = ["red", "green", "blue"]

        if not exclude_nir:
            features_names.append("nir")

        input_tensor = []
        for feature_name in features_names:
            with rasterio.open(self.generated_dataset_path / f"{sample_num}_{feature_name}.tif") as f:
                input_tensor.append(veg_index.preprocess_band(f.read(1)))

        ground_truth_tensor = []
        with rasterio.open(self.generated_dataset_path / f"{sample_num}_mask.tif") as f:
            ground_truth_tensor.append(f.read(1))

        if not exclude_fMASK:
            input_tensor.append(self.create_forest_mask(sample_num))

        input_tensor = np.array(input_tensor)
        loaded_model = load_model(model, model_path)

        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()

            predict_mask = evaluate(loaded_model, input_tensor)

            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = sum(times) / num_runs
        print(f"Average inference time: {avg_time:.6f} seconds")

    def model_summary_structure(self, model, name="model", exclude_nir=False, exclude_fMASK=False):
        sample, mask = next(self.get_next_generated_sample(exclude_nir=exclude_nir, exclude_fMASK=exclude_fMASK))
        summary(model, sample.shape, batch_size=1, device="cpu")
        output = model(model.prepare_input(sample))
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.format = "png"
        dot.render(name)
        model_graph = draw_graph(model, input_size=sample.shape, expand_nested=True)
        model_graph.visual_graph
