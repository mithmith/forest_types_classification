import os
import re
import shutil
from pathlib import Path

import rasterio
from loguru import logger
from skimage.transform import resize

import app.utils.veg_index as veg_index

os.environ["GDAL_DATA"] = os.environ["CONDA_PREFIX"] + r"\Library\share\gdal"
os.environ["PROJ_LIB"] = os.environ["CONDA_PREFIX"] + r"\Library\share"
os.environ["GDAL_DRIVER_PATH"] = os.environ["CONDA_PREFIX"] + r"\Library\lib\gdalplugins"


def move_files_to_root(root_path: Path):
    """
    Перемещает все файлы из вложенных папок в корневую папку, заменяя существующие файлы, и удаляет пустые папки.

    :param root_path: Path объект, указывающий на корневую папку.
    """
    if not root_path.is_dir():
        logger.info(f"Путь {root_path} не является директорией!")
        return

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Пропускаем корневую папку
        if Path(dirpath) == root_path:
            continue

        for filename in filenames:
            source = Path(dirpath) / filename
            destination = root_path / filename

            # Перемещаем файл, заменяя существующий, если он есть
            shutil.move(str(source), str(destination))
            logger.info(f"Перемещён файл {source} -> {destination}")

        # Удаляем пустую папку
        for dirname in dirnames:
            dir_to_remove = Path(dirpath) / dirname
            if not os.listdir(dir_to_remove):  # Проверяем, что папка пуста
                dir_to_remove.rmdir()
                logger.info(f"Удалена пустая папка {dir_to_remove}")

    # Удаляем пустые папки в корне
    for dirpath, dirnames, _ in os.walk(root_path, topdown=False):
        for dirname in dirnames:
            dir_to_remove = Path(dirpath) / dirname
            if not os.listdir(dir_to_remove):
                dir_to_remove.rmdir()
                logger.info(f"Удалена пустая папка {dir_to_remove}")


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


def save_new_band(bands_path: Path):
    bands_regex_list = {
        "red": r"(\w*)_(\d{8}T\d*)_(B04|B04_10m)\.jp2",
        "swir1": r"(\w*)_(\d{8}T\d*)_(B11|B11_20m)\.jp2",
        "swir2": r"(\w*)_(\d{8}T\d*)_(B12|B12_20m)\.jp2",
    }

    red_image_shape = None

    for band_key, band_regex in bands_regex_list.items():
        band_path = get_band_images(bands_path, band_regex)
        if bands_path.exists():
            if red_image_shape is None:
                _, (red_image_shape, _, _) = veg_index.read_band(band_path)
                with rasterio.open(band_path) as src:
                    profile = src.profile
            else:
                band, _ = veg_index.read_band(band_path)
                band_resized = resize(band, red_image_shape)
                if "20m" in str(band_path):
                    output_path = str(band_path).replace("20m", "10m")
                else:
                    output_path = str(band_path)
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(band_resized, 1)


if __name__ == "__main__":
    # Получаем все папки со снимками Sentinel-2 и перемещаем их в корневую папку для использования
    root_folder = Path("G:/Orni_forest/forest_changes_dataset/images_test/")
    for folder in root_folder.glob("*.SAFE/"):
        move_files_to_root(folder)
        logger.info(f"Preprocessing B11/B12: {folder}")
        save_new_band(Path(folder))
