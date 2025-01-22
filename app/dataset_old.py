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
from pyproj import CRS as pyproj_crs
from rasterio.features import Affine, rasterize
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_geom
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy.ndimage import rotate
from shapely.geometry import Polygon, shape
from tqdm import tqdm

from app.services.base import BoundingBox
from app.utils.veg_index import evi_formula, ndvi_formula, preprocess_band


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
        self.datset_length = 0

    def __len__(self):
        return self.datset_length

    @staticmethod
    def get_masks_info(geojson_path: Path) -> tuple[list[dict[str, Any]], str]:
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data: dict = json.load(f)

        features_data: list[dict[str, Any]] = []

        # Извлечение CRS и преобразование в стандартный формат
        crs_raw = geojson_data.get("crs", {}).get("properties", {}).get("name", "")
        try:
            crs = pyproj_crs.from_user_input(crs_raw).to_string()
        except Exception:
            crs = "EPSG:4326"  # Если определить CRS не удалось, возвращаем WGS84

        # Обработка каждого feature в массиве "features"
        for feature in geojson_data.get("features", []):
            if feature.get("type") != "Feature":
                continue

            properties = feature.get("properties", {})
            feature_info = {
                "img1": properties.get("ID_B_S", "") + ".SAFE",  # 1 изображение - before
                "img2": properties.get("ID_A_S", "") + ".SAFE",  # 2 изображение - after
            }

            # Обработка геометрии
            geometry = feature.get("geometry", {})
            if geometry.get("type") == "Polygon":
                feature_info["polygon"] = geometry.get("coordinates")
            elif geometry.get("type") == "MultiPolygon":
                feature_info["polygon"] = geometry.get("coordinates", [])[0]  # Только первый полигон

            features_data.append(feature_info)

        return features_data, crs

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
    def get_polygons_for_images(img1_name: str, img2_name: str, masks_data: list[dict]) -> list[Polygon]:
        # Собираем общую маску всех полигонов с вырубками (отмечаются в forest_mask["img2"])
        # т.е. вырубка найдена на одном из снимков img1_name или img2_name - для формирования общей маски.
        # собираем массив полигонов для img1 -> list[coords]
        # собираем массив полигонов для img2 -> list[coords]
        # объединяем массивы - extend - получаем общий массив всех полигонов для img1 и img2
        polygons = []
        for forest_mask in masks_data:
            # Полигон с вырубкой найден на одном из снимков
            if forest_mask["img2"] == img1_name or forest_mask["img2"] == img2_name:
                polygons.append(Polygon(forest_mask["polygon"][0]))
        return polygons

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

    def create_polygons_mask(self, tif1_path: Path, coords_list: list, coords_crs) -> tuple[np.ndarray, Affine, str]:
        # Открываем tif1 для получения трансформации и системы координат
        with rasterio.open(tif1_path) as src:
            img_width, img_height = src.width, src.height
            transform_matrix = src.transform
            target_crs = src.crs

        # Преобразование полигонов в CRS файла tif1
        transformed_polygons = []
        for polygon in coords_list:
            transformed_geom = transform_geom(coords_crs, coords_crs, shape(polygon).__geo_interface__)
            transformed_polygons.append(transformed_geom)

        # Создание бинарной маски размером с изображение
        mask = np.zeros((img_height, img_width), dtype=np.int8)

        # Растризация полигонов на маске
        mask = rasterize(
            [(shape(poly), 1) for poly in transformed_polygons],
            out_shape=(img_height, img_width),
            transform=transform_matrix,
            fill=0,
            dtype=np.uint8,
            all_touched=True,  # Учитывает все пиксели, которых касается полигон
        )
        # self.save_mask(
        #     mask, img_height, img_width, transform_matrix, target_crs, self.generated_dataset_path / "mask.tif"
        # )
        return mask, transform_matrix, target_crs

    def generate_dataset_images(self, polygons: list[Polygon], coords_crs: str, dir_with_warped_bands: Path) -> int:
        logger.debug(f"\ndir_with_warped_bands={dir_with_warped_bands}\ncoords_crs={coords_crs}")
        logger.debug(f"Polygons: {len(polygons)}")
        output_dir_path = self.generated_dataset_path
        tif1_path, tif2_path = (
            dir_with_warped_bands / f"1_red.tif",
            dir_with_warped_bands / f"2_red.tif",
        )

        img_width, img_height = self.get_min_images_shape(tif1_path, tif2_path)
        union_mask, trans_matr, trgt_crs = self.create_polygons_mask(tif1_path, polygons, coords_crs)
        logger.debug(f"union_mask shape: {union_mask.shape}")
        num_samples = int((img_width // self.image_shape[0] + 1) * (img_height // self.image_shape[1] + 1) * 2)
        total_samples: int = 0

        for n in range(num_samples):
            box = self.get_random_bbox(img_width, img_height, self.image_shape)
            mask_region = union_mask[box.miny : box.maxy, box.minx : box.maxx]
            # if np.count_nonzero(mask_region) < 1000:
            #     # logger.info("Bounding box is empty, trying another one...")
            #     continue

            rnd_num = random.randint(100000, 999999)
            crop_mask_path = output_dir_path / f"{rnd_num}_mask.tif"
            window = Window(box.minx, box.miny, box.width, box.height)
            cropped_transform_matrix = window_transform(window, trans_matr)
            self.save_mask(mask_region, box.height, box.width, cropped_transform_matrix, trgt_crs, crop_mask_path)
            for band_key in self.bands_regex_list.keys():
                tif1_path, tif2_path = (
                    dir_with_warped_bands / f"1_{band_key}.tif",
                    dir_with_warped_bands / f"2_{band_key}.tif",
                )
                logger.debug(
                    f"band_key={band_key}, img_width={img_width}, img_height={img_height}, num_samples={n}/{num_samples}"
                )

                crop_output_path1 = output_dir_path / f"{rnd_num}_{band_key}_1.tif"
                crop_output_path2 = output_dir_path / f"{rnd_num}_{band_key}_2.tif"
                self.save_cropped_region(tif1_path, box, crop_output_path1)
                self.save_cropped_region(tif2_path, box, crop_output_path2)
            total_samples += 1
        logger.debug(f"Сгенерировано {total_samples} сэмплов.")
        return total_samples

    def generate_dataset(self, num_samples: int = 100):
        if self.generated_dataset_path is None:
            raise ValueError("Generate dataset path is required")
        if not self.geojson_files:
            raise ValueError("GeoJSON files are required")

        total = 0
        for geojson_path in self.geojson_files:
            forest_masks_data, masks_crs = self.get_masks_info(geojson_path)
            logger.debug(f"Количество полигонов: {len(forest_masks_data)}")

            for _ in tqdm(range(num_samples)):
                forest_mask = random.choice(forest_masks_data)
                img1_dir_path = self.sentinel_root / Path(forest_mask["img1"])
                img2_dir_path = self.sentinel_root / Path(forest_mask["img2"])

                if img1_dir_path.exists() and img2_dir_path.exists():
                    logger.info("Папки со снимками найдены!")
                else:
                    logger.error("Не найдены папки со снимками!")
                    continue

                if self.check_all_bands_exist(img1_dir_path, img2_dir_path):
                    logger.info("Все необходимые бэнды на месте!")
                else:
                    logger.error("Не найдены все банды!")
                    continue

                with TemporaryDirectory() as tmpdirname:
                    self.warp_all_bands(img1_dir_path, img2_dir_path, Path(tmpdirname))
                    polygons = self.get_polygons_for_images(forest_mask["img1"], forest_mask["img2"], forest_masks_data)
                    total += self.generate_dataset_images(polygons, masks_crs, Path(tmpdirname))

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
        red_tif_files = list(self.generated_dataset_path.glob("*_red_*.tif"))
        self.datset_length = len(red_tif_files)
        random.shuffle(red_tif_files)

        for filename in red_tif_files:
            # Извлекаем номер выборки и путь до маски
            n = filename.stem.split("_")[0]
            mask_path = self.generated_dataset_path / f"{n}_mask.tif"
            bands_path_1, bands_path_2 = [], []

            for band_key in self.bands_regex_list.keys():
                bands_path_1.append(self.generated_dataset_path / f"{n}_{band_key}_1.tif")
                bands_path_2.append(self.generated_dataset_path / f"{n}_{band_key}_2.tif")

            try:
                # Загрузка всех бэндов в многослойный массив для bands_path_1
                band_data_1: list[np.ndarray] = []
                for band_path in bands_path_1:
                    with rasterio.open(band_path) as src:
                        band = preprocess_band(src.read(1))
                        noisy_band = self.add_salt_and_pepper_noise(band, salt_percent=0.01, pepper_percent=0.01)
                        noisy_band = self.add_gaussian_noise(noisy_band)
                        band_data_1.append(noisy_band)  # Чтение первого слоя
                band_data_1.append(self.create_forest_mask(bands_path_1[3], bands_path_1[0], bands_path_1[2]))

                # Загрузка всех бэндов в многослойный массив для bands_path_2
                band_data_2: list[np.ndarray] = []
                for band_path in bands_path_2:
                    with rasterio.open(band_path) as src:
                        band = preprocess_band(src.read(1))
                        noisy_band = self.add_salt_and_pepper_noise(band, salt_percent=0.01, pepper_percent=0.01)
                        noisy_band = self.add_gaussian_noise(noisy_band)
                        band_data_2.append(noisy_band)  # Чтение первого слоя
                band_data_2.append(self.create_forest_mask(bands_path_2[3], bands_path_2[0], bands_path_2[2]))

                # Загрузка маски
                with rasterio.open(mask_path) as src:
                    mask_data = src.read(1)  # Чтение первого слоя маски

                if verbose:
                    # Визуализация первого канала и маски до и после трансформаций
                    plt.figure(figsize=(16, 16))

                    # Отображение каналов band_data_2
                    for i in range(3):
                        plt.subplot(2, 4, i + 1)
                        plt.imshow(band_data_2[i] * 255, cmap="gray")
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
            except Exception as e:
                logger.error(e)
                continue

            yield band_data_1, band_data_2, mask_data
