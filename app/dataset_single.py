import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from loguru import logger
from osgeo import gdal
from rasterio.features import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy.ndimage import rotate
from tqdm import tqdm
import os
from datetime import datetime

from app.services.base import BoundingBox
from app.utils.veg_index import evi_formula, ndvi_formula, preprocess_band
import utils.geo_mask as geo_mask


class ForestTypesDataset:
    def __init__(self, geojson_masks_dir: Path, sentinel_root_path: Path, dataset_path: Path | None = None) -> None:
        self.image_shape = (512, 512)
        self.bands_regex_list = {
            "red": r"(\w*)_(\d{8}T\d*)_(B04|B04_10m)\.jp2",
            "green": r"(\w*)_(\d{8}T\d*)_(B03|B03_10m)\.jp2",
            "blue": r"(\w*)_(\d{8}T\d*)_(B02|B02_10m)\.jp2",
            "nir": r"(\w*)_(\d{8}T\d*)_(B08|B08_10m)\.jp2",
        }
        self.sentinel_root = sentinel_root_path
        self.generated_dataset_path = dataset_path
        self.geojson_files = list(geojson_masks_dir.glob("*.geojson"))
        self.images_files = [f for f in os.listdir(sentinel_root_path)]
        self.dataset_length = 0

    def __len__(self):
        return self.dataset_length

    def find_geojson_path_for_img(self, img: str):
        first_part = img.split('_')[2]
        second_part = img.split('_')[5]
        for s in self.geojson_files:
            if first_part in str(s) and second_part in str(s):
                mask_path = Path(s)
                return mask_path
        return None

    def get_info_from_img(self, img_name: str):
        first_part = img_name.split('_')[2]
        second_part = img_name.split('_')[5]

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
                profile.update({"height": box.maxy - box.miny, "width": box.maxx - box.minx, "transform": transform,
                                "driver": "GTiff", "compress": "lzw", "tiled": True})

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
            for img_file in self.images_files:
                geojson_mask_path = self.find_geojson_path_for_img(img_file)
                if geojson_mask_path is None:
                    logger.info(f'Mask not found for image {img_file}, skipping.')
                    continue

                transform_matrix, img_width, img_height, crs = self.get_info_from_img(img_file)
                with open(geojson_mask_path) as f:
                    geojson_dict = json.load(f)

                mask = geo_mask.mask_from_geojson(geojson_dict, (img_height, img_width), transform_matrix)

                generated_samples = 0
                while generated_samples < samples_per_image:
                    box = self.get_random_bbox(img_width, img_height, self.image_shape)
                    mask_region = mask[box.miny: box.maxy, box.minx: box.maxx]
                    if np.count_nonzero(mask_region) <= 1000:
                        continue

                    rnd_num = random.randint(100000, 999999)
                    crop_mask_path = self.generated_dataset_path / f"{rnd_num}_mask.tif"
                    cropped_transform_matrix = window_transform(
                        Window(box.minx, box.miny, box.width, box.height), transform_matrix
                    )

                    self.save_mask(mask_region, box.height, box.width, cropped_transform_matrix, crs, crop_mask_path)

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
        Находит и возвращает пути к файлам заданного бэнда в папках img1_path и img2_path.

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

    def create_forest_mask(self, nir_path: Path, red_path: Path, blue_path: Path) -> np.ndarray:
        with rasterio.open(nir_path) as nir, rasterio.open(red_path) as red, rasterio.open(blue_path) as blue:
            nir_band = preprocess_band(nir.read(1))
            red_band = preprocess_band(red.read(1))
            blue_band = preprocess_band(blue.read(1))
        ndvi = ndvi_formula(nir_band, red_band)
        evi = evi_formula(nir_band, red_band, blue_band)
        return np.array((ndvi > 0.5) & (evi > 0.5), dtype=np.uint8)

    def get_next_generated_sample(
        self, verbose: bool = False, exclude_nir=True
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
                if 'nir' in str(band_key) and exclude_nir:
                    continue
                band_file = self.generated_dataset_path / f"{sample_id}_{band_key}.tif"
                if band_file.exists():
                    with rasterio.open(band_file) as band_src:
                        band = preprocess_band(band_src.read(1))
                        band = self.add_salt_and_pepper_noise(band, salt_percent=0.01, pepper_percent=0.01)
                        band = self.add_gaussian_noise(band)
                        band_data.append(band)
            # band_data.append(self.create_forest_mask(band_data[3], band_data[0], band_data[2]))
                else:
                    logger.warning(f"Missing band file: {band_file}, skipping this sample.")
                    continue

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
