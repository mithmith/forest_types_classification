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
    def get_min_images_shape(tif1_path: Path, tif2_path: Path) -> tuple[int, int]:
        with rasterio.open(tif1_path, "r") as f:
            img_width, img_height = f.shape
        with rasterio.open(tif2_path, "r") as f:
            img_width_2, img_height_2 = f.shape
        return min(img_width, img_width_2), min(img_height, img_height_2)

    @staticmethod
    def get_band_images(img1_path: Path, img2_path: Path, band_regex: str) -> tuple[Path, Path]:
        """
        Находит и возвращает пути к файлам заданного бэнда в папках img1_path и img2_path.

        Args:
            img1_path (Path): Путь к первой папке со снимком.
            img2_path (Path): Путь ко второй папке со снимком.
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
        tif1_path = find_band_file(img1_path)
        tif2_path = find_band_file(img2_path)

        return tif1_path, tif2_path

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
    def save_cropped_region(tif_path: Path, box: BoundingBox, output_path: Path):
        with rasterio.open(tif_path) as src:
            # Создаем окно для вырезки области по координатам box
            window = Window(box.minx, box.miny, box.maxx - box.minx, box.maxy - box.miny)
            transform = src.window_transform(window)

            # Читаем данные окна
            cropped_data: np.ndarray = src.read(1, window=window)

            # Сохраняем вырезанные данные в новый tif файл
            profile: dict = src.profile
            profile.update({"height": box.maxy - box.miny, "width": box.maxx - box.minx, "transform": transform})

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(cropped_data, 1)

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

    def check_all_bands_exist(self, img1_path: Path, img2_path: Path) -> bool:
        for band_key in self.bands_regex_list.keys():
            try:
                _, _ = self.get_band_images(img1_path, img2_path, band_regex=self.bands_regex_list[band_key])
            except:
                logger.error(f"Band {band_key} not found in images: {img1_path}, {img2_path}")
                return False
        return True

    def process_band(self, band_key: str, band_regex: str, img1_dir_path: Path, img2_dir_path: Path, output_dir: Path):
        """
        Обрабатывает один бэнд, вырезает пересечения из img1 и img2 и сохраняет их в output_dir.
        """
        # Получаем пути к изображениям бэндов
        tif1_path, tif2_path = self.get_band_images(img1_dir_path, img2_dir_path, band_regex)

        # Открываем оба изображения для работы с ними
        with rasterio.open(tif1_path) as src1, rasterio.open(tif2_path) as src2:
            # Находим трансформацию и параметры для приведения к общим координатам
            transform1, width1, height1 = calculate_default_transform(
                src1.crs, src1.crs, src1.width, src1.height, *src1.bounds
            )
            profile1 = src1.profile
            profile1.update({"transform": transform1, "width": width1, "height": height1})

            transform2, width2, height2 = calculate_default_transform(
                src2.crs, src2.crs, src2.width, src2.height, *src2.bounds
            )
            profile2 = src2.profile
            profile2.update({"transform": transform2, "width": width2, "height": height2})

            # Создаём выходные файлы для пересечений
            intersect_path1 = output_dir / f"1_{band_key}.tif"
            intersect_path2 = output_dir / f"2_{band_key}.tif"

            with rasterio.open(intersect_path1, "w", **profile1) as dst1:
                reproject(
                    source=rasterio.band(src1, 1),
                    destination=rasterio.band(dst1, 1),
                    src_transform=src1.transform,
                    src_crs=src1.crs,
                    dst_transform=transform1,
                    dst_crs=src1.crs,
                    resampling=Resampling.nearest,
                )

            with rasterio.open(intersect_path2, "w", **profile2) as dst2:
                reproject(
                    source=rasterio.band(src2, 1),
                    destination=rasterio.band(dst2, 1),
                    src_transform=src2.transform,
                    src_crs=src2.crs,
                    dst_transform=transform2,
                    dst_crs=src2.crs,
                    resampling=Resampling.nearest,
                )

        print(f"Intersected band {band_key} saved at {intersect_path1} and {intersect_path2}")

    def process_band_gdal(
        self, band_key: str, band_regex: str, img1_dir_path: Path, img2_dir_path: Path, output_dir: Path
    ):
        """
        Обрабатывает один бэнд, вырезает пересечения из img1 и img2 и сохраняет их в output_dir.
        """
        tif1_path, tif2_path = self.get_band_images(img1_dir_path, img2_dir_path, band_regex)

        # Создаём выходные пути
        intersect_path1 = output_dir / f"1_{band_key}.tif"
        intersect_path2 = output_dir / f"2_{band_key}.tif"

        # Вырезаем пересечения и сохраняем в временную папку
        src = gdal.Open(str(tif2_path))
        ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
        lrx = ulx + (src.RasterXSize * xres)
        lry = uly + (src.RasterYSize * yres)
        src = None

        # Обрезка первого файла
        gdal.Warp(str(intersect_path1), str(tif1_path), format="GTiff", outputBounds=(ulx, lry, lrx, uly), dstNodata=0)

        src = gdal.Open(str(tif1_path))
        ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
        lrx = ulx + (src.RasterXSize * xres)
        lry = uly + (src.RasterYSize * yres)
        src = None

        # Обрезка второго файла
        gdal.Warp(str(intersect_path2), str(tif2_path), format="GTiff", outputBounds=(ulx, lry, lrx, uly), dstNodata=0)
        print(f"Intersected band {band_key} saved at {intersect_path1} and {intersect_path2}")

    def warp_all_bands(self, img1_dir_path: Path, img2_dir_path: Path, output_dir: Path):
        """
        Вырезает пересечения всех бэндов из img1_dir_path и img2_dir_path и сохраняет их в output_dir.

        Args:
            img1_dir_path (Path): Путь к первой папке с изображениями.
            img2_dir_path (Path): Путь ко второй папке с изображениями.
            output_dir (Path): Путь к папке для сохранения результирующих файлов.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        gdal.UseExceptions()
        gdal.SetConfigOption("GDAL_SKIP", "JPEG2000")
        gdal.SetConfigOption("GDAL_ENABLE_DEPRECATED_DRIVER_JPEG2000", "NO")

        # Запускаем обработку каждого бэнда параллельно
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self.process_band_gdal, band_key, band_regex, img1_dir_path, img2_dir_path, output_dir)
                for band_key, band_regex in self.bands_regex_list.items()
            ]

            # Ожидание завершения всех заданий
            for future in futures:
                future.result()

        # for band_key, band_regex in self.bands_regex_list.items():
        #     self.process_band_gdal(band_key, band_regex, img1_dir_path, img2_dir_path, output_dir)

    def generate_dataset_images(self, difference_mask, trans_matr, coords_crs: str, dir_with_warped_bands: Path) -> int:
        logger.debug(f"\ndir_with_warped_bands={dir_with_warped_bands}\ncoords_crs={coords_crs}")
        output_dir_path = self.generated_dataset_path
        tif1_path, tif2_path = (
            dir_with_warped_bands / f"1_red.tif",
            dir_with_warped_bands / f"2_red.tif",
        )

        img_width, img_height = self.get_min_images_shape(tif1_path, tif2_path)

        num_samples = int((img_width // self.image_shape[0] + 1) * (img_height // self.image_shape[1] + 1) * 2)
        total_samples: int = 0

        for _ in tqdm(range(num_samples), position=0, leave=True):
            box = self.get_random_bbox(img_width, img_height, self.image_shape)
            mask_region = difference_mask[box.miny: box.maxy, box.minx: box.maxx]
            if np.count_nonzero(mask_region) < 1000:
                # logger.info("Bounding box is empty, trying another one...")
                continue

            rnd_num = random.randint(100000, 999999)
            crop_mask_path = output_dir_path / f"{rnd_num}_mask.tif"
            window = Window(box.minx, box.miny, box.width, box.height)
            cropped_transform_matrix = window_transform(window, trans_matr)

            self.save_mask(mask_region, box.height, box.width, cropped_transform_matrix, coords_crs, crop_mask_path)
            for band_key in self.bands_regex_list.keys():
                tif1_path, tif2_path = (
                    dir_with_warped_bands / f"1_{band_key}.tif",
                    dir_with_warped_bands / f"2_{band_key}.tif",
                )

                crop_output_path1 = output_dir_path / f"{rnd_num}_{band_key}_1.tif"
                crop_output_path2 = output_dir_path / f"{rnd_num}_{band_key}_2.tif"
                self.save_cropped_region(tif1_path, box, crop_output_path1)
                self.save_cropped_region(tif2_path, box, crop_output_path2)
            total_samples += 1
        logger.debug(f"Сгенерировано {total_samples} сэмплов.")
        return total_samples

    def generate_imgs_pairs(self,):
        grouped_imgs = {}
        for file in self.images_files:
            key = file.split('_')[5]  # Extract the key part
            if key not in grouped_imgs:
                grouped_imgs[key] = []
            grouped_imgs[key].append(file)

        sorted_pairs = {}
        for key, group in grouped_imgs.items():
            sorted_group = sorted(group, key=lambda s: datetime.strptime(s.split('_')[2], '%Y%m%dT%H%M%S'))
            sorted_pairs[key] = sorted_group

        imgs_pairs = []

        # Generate all possible pairs for each group
        for key, sorted_group in sorted_pairs.items():
            if len(sorted_group) < 2:
                logger.info(f"Not enough images to form pairs for key: {key}")
                continue

            # Form pairs: first with older date, second with newer date
            for i in range(len(sorted_group)):
                for j in range(i + 1, len(sorted_group)):
                    imgs_pairs.append((sorted_group[i], sorted_group[j]))

        return imgs_pairs

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

    def generate_dataset(self, num_samples: int = 100):
        if self.generated_dataset_path is None:
            raise ValueError("Generate dataset path is required")
        if not self.geojson_files:
            raise ValueError("GeoJSON files are required")

        total = 0
        for img_before, img_after in self.generate_imgs_pairs():
            geojson_before_mask_path = self.find_geojson_path_for_img(img_before)
            geojson_after_mask_path = self.find_geojson_path_for_img(img_after)
            if geojson_before_mask_path is None:
                logger.info('Mask_before not found, skipping to next pair')
                continue
            if geojson_after_mask_path is None:
                logger.info('Mask_after not found, skipping to next pair')
                continue

            transform_matrix, img_width, img_height, crs = self.get_info_from_img(img_before)
            with open(geojson_before_mask_path) as f:
                dict_before = json.load(f)
            mask_before = geo_mask.mask_from_geojson(dict_before, (img_height, img_width), transform_matrix)

            transform_matrix, img_width, img_height, crs = self.get_info_from_img(img_after)
            with open(geojson_after_mask_path) as f:
                dict_after = json.load(f)
            mask_after = geo_mask.mask_from_geojson(dict_after, (img_height, img_width), transform_matrix)

            difference_mask = mask_after & (~mask_before)

            for _ in tqdm(range(num_samples)):
                img_before_dir_path = self.sentinel_root / img_before
                img_after_dir_path = self.sentinel_root / img_after

                if img_before_dir_path.exists() and img_after_dir_path.exists():
                    logger.info("Папки со снимками найдены!")
                else:
                    logger.error("Не найдены папки со снимками!")
                    continue

                if self.check_all_bands_exist(img_before_dir_path, img_after_dir_path):
                    logger.info("Все необходимые бэнды на месте!")
                else:
                    logger.error("Не найдены все банды!")
                    continue

                with TemporaryDirectory() as tmpdirname:
                    self.warp_all_bands(img_before_dir_path, img_after_dir_path, Path(tmpdirname))
                    total += self.generate_dataset_images(difference_mask, transform_matrix, crs, Path(tmpdirname))

        logger.info(f"Всего сгенерировано {total} файлов")

    def create_forest_mask(self, nir_path: Path, red_path: Path, blue_path: Path) -> np.ndarray:
        with rasterio.open(nir_path) as nir, rasterio.open(red_path) as red, rasterio.open(blue_path) as blue:
            nir_band = preprocess_band(nir.read(1))
            red_band = preprocess_band(red.read(1))
            blue_band = preprocess_band(blue.read(1))
        ndvi = ndvi_formula(nir_band, red_band)
        evi = evi_formula(nir_band, red_band, blue_band)
        return np.array((ndvi > 0.5) & (evi > 0.5), dtype=np.uint8)

    def get_next_generated_sample(
        self, verbose: bool = False
    ) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
        red_tif_files = list(self.generated_dataset_path.glob("*_red_2.tif"))
        self.dataset_length = len(red_tif_files)
        random.shuffle(red_tif_files)

        for filename in red_tif_files:
            # Извлекаем номер выборки и путь до маски
            n = filename.stem.split("_")[0]
            mask_path = self.generated_dataset_path / f"{n}_mask.tif"
            bands_path_1, bands_path_2 = [], []

            for band_key in self.bands_regex_list.keys():
                if band_key == 'nir':
                    continue
                bands_path_1.append(self.generated_dataset_path / f"{n}_{band_key}_1.tif")
                bands_path_2.append(self.generated_dataset_path / f"{n}_{band_key}_2.tif")

            # Загрузка всех бэндов в многослойный массив для bands_path_1
            band_data_1: list[np.ndarray] = []
            for band_path in bands_path_1:
                if 'nir' in str(band_path):
                    continue
                with rasterio.open(band_path) as src:
                    band = preprocess_band(src.read(1))
                    noisy_band = self.add_salt_and_pepper_noise(band, salt_percent=0.01, pepper_percent=0.01)
                    noisy_band = self.add_gaussian_noise(noisy_band)
                    band_data_1.append(noisy_band)  # Чтение первого слоя
            # band_data_1.append(self.create_forest_mask(bands_path_1[3], bands_path_1[0], bands_path_1[2]))

            # Загрузка всех бэндов в многослойный массив для bands_path_2
            band_data_2: list[np.ndarray] = []
            for band_path in bands_path_2:
                if 'nir' in str(band_path):
                    continue
                with rasterio.open(band_path) as src:
                    band = preprocess_band(src.read(1))
                    noisy_band = self.add_salt_and_pepper_noise(band, salt_percent=0.01, pepper_percent=0.01)
                    noisy_band = self.add_gaussian_noise(noisy_band)
                    band_data_2.append(noisy_band)  # Чтение первого слоя
            # band_data_2.append(self.create_forest_mask(bands_path_2[3], bands_path_2[0], bands_path_2[2]))

            # Загрузка маски
            with rasterio.open(mask_path) as src:
                mask_data = src.read(1)  # Чтение первого слоя маски

            if verbose:
                # Визуализация первого канала и маски до и после трансформаций
                plt.figure(figsize=(16, 16))

                # Отображение каналов band_data_2
                for i in range(3):
                    plt.subplot(2, 4, i + 1)
                    plt.imshow(band_data_2[i]*255, cmap="gray")
                    plt.title(f"Original Band {i + 1}")

                plt.subplot(2, 4, 4)
                plt.imshow(mask_data, cmap="gray")
                plt.title("Original Mask")
            # Добавляем случайные повороты и отражения
            band_data_1, band_data_2, mask_data = self.add_random_rotation_and_flip(
                band_data_1, band_data_2, mask_data
            )

            if verbose:
                # Отображение каналов band_data_2
                for i in range(3):
                    plt.subplot(2, 4, i + 5)
                    data = (band_data_2[i] - band_data_2[i].mean()) / band_data_2[i].std()
                    data = (data - data.min()) / (data.max() - data.min())
                    plt.imshow(data, cmap="gray")
                    plt.title(f"Original Band {i + 1}")

                plt.subplot(2, 4, 8)
                plt.imshow(mask_data, cmap="gray")
                plt.title("Transformed Mask")

                plt.show()

            band_data_1 = np.stack(band_data_1, axis=0)  # Формируем многослойный массив 1
            band_data_2 = np.stack(band_data_2, axis=0)  # Формируем многослойный массив 2
            # except Exception as e:
            #     logger.error(e)
            #     continue

            yield band_data_1, band_data_2, mask_data
