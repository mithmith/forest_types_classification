from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from loguru import logger
from skimage.transform import resize

from app.utils.constants import INDEX_BAND_MAPPER, INDEX_BANDS


def blue_formula(blue: np.ndarray) -> np.ndarray:
    return blue


def green_formula(green: np.ndarray) -> np.ndarray:
    return green


def red_formula(red: np.ndarray) -> np.ndarray:
    return red


def nir_formula(nir: np.ndarray) -> np.ndarray:
    return nir


def swir1_formula(swir1: np.ndarray) -> np.ndarray:
    return swir1


def swir2_formula(swir2: np.ndarray) -> np.ndarray:
    return swir2


def ndvi_formula(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.clip((nir - red) / (nir + red), -1, 1)


def evi_formula(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        znam = nir + 6 * red - 7.5 * blue + 1
        return np.clip(2.5 * (nir - red) / znam, -1, 1)


def ndwi_formula(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((green - nir) / (green + nir), -1, 1)


def mndwi_formula(green: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip((green - swir2) / (green + swir2), -1, 1)


def ndmi_formula(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.clip((nir - swir1) / (nir + swir1), -1, 1)


def nbr_formula(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.clip((nir - swir2) / (nir + swir2), -1, 1)


def preprocess_band(band: np.ndarray) -> np.ndarray:
    return np.clip(np.array(band / 10000 - 0.1, dtype=np.float32), 0, 1)


INDEX_FORMULAS = {
    "blue": blue_formula,
    "green": green_formula,
    "red": red_formula,
    "nir": nir_formula,
    "swir1": swir1_formula,
    "swir2": swir2_formula,
    "ndvi": ndvi_formula,
    "evi": evi_formula,
    "ndwi": ndwi_formula,
    "mndwi": mndwi_formula,
    "ndmi": ndmi_formula,
    "nbr": nbr_formula,
}


def filter_noise_on_mask(binary_mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Применяет морфологические операции для сглаживания бинарной маски:
    заполнение пропусков и удаление шума.

    :param binary_mask: Входная бинарная маска (numpy.ndarray)
    :param kernel_size: Размер структурирующего элемента (нечетное число)
    :return: Сглаженная бинарная маска (numpy.ndarray)
    """
    # Создаем структурирующий элемент
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Применяем дилатацию для заполнения дыр
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=3)

    # Применяем эрозию для удаления шумов
    smoothed_mask = cv2.erode(dilated_mask, kernel, iterations=3)

    return smoothed_mask


def get_bands_path_list(sentinel_image_dir_path: Path, level: int = 2) -> dict[str, Path]:
    """
    Инициализация путей до файлов спутникового снимка для последующего анализа.

    Находит файлы спектральных каналов в указанной директории.

    Args:
    - sentinel_image_dir_path (Path): Путь до папки с файлами спутникового снимка.
    - level: int: Уровень обработки снимка: L2A или L1C
    """
    output_image_data: dict[str, Path] = {}
    if sentinel_image_dir_path:
        for file in sentinel_image_dir_path.iterdir():
            if file.name.endswith(".tif") or file.name.endswith(".jp2"):
                parts = file.name.split("_")

                # Определение бэнда
                if level == 2 and parts[-1].split(".")[0] in ["10m", "20m", "60m"]:
                    band_part = parts[-2]  # Бэнд перед разрешением
                    scale_part = parts[-1].split(".")[0]  # Пространственное разрешение
                else:
                    band_part = parts[-1].split(".")[0]  # Последняя часть для L1C
                    scale_part = None

                # Определение файлов по бэнду и разрешению
                if band_part == "B02" and (scale_part == "10m" or level == 1):
                    output_image_data["blue"] = file
                if band_part == "B03" and (scale_part == "10m" or level == 1):
                    output_image_data["green"] = file
                if band_part == "B04" and (scale_part == "10m" or level == 1):
                    output_image_data["red"] = file
                if band_part == "B05" and (scale_part == "20m" or level == 1):
                    output_image_data["rededge1"] = file
                if band_part == "B06" and (scale_part == "20m" or level == 1):
                    output_image_data["rededge2"] = file
                if band_part == "B07" and (scale_part == "20m" or level == 1):
                    output_image_data["rededge3"] = file
                if band_part == "B08" and (scale_part == "10m" or level == 1):
                    output_image_data["nir"] = file
                if band_part == "B11" and (scale_part == "20m" or level == 1):
                    output_image_data["swir1"] = file
                if band_part == "B12" and (scale_part == "20m" or level == 1):
                    output_image_data["swir2"] = file
                if band_part == "CLDPRB" and scale_part == "20m":
                    output_image_data["qa"] = file
    return output_image_data


def read_band(file_path: Path, target_type: str = "float64") -> tuple[np.ndarray, tuple[tuple, dict, Any]]:
    # logger.debug(f"Reading image from {file_path}")
    if file_path.name.endswith(".jp2"):
        with rasterio.open(file_path, driver="JP2OpenJPEG") as src:
            return (src.read(1).astype(target_type), ((src.width, src.height), src.transform, src.crs))
    with rasterio.open(file_path) as src:
        return (src.read(1).astype(target_type), ((src.width, src.height), src.transform, src.crs))


def calculate_image_data(
    bands_path_list: dict[str, Path], bands_list: list[str], level: int
) -> tuple[np.ndarray, tuple]:
    bands_data: dict[str, np.ndarray] = {}
    image_shape, transform_matrix, crs = None, None, None

    # Загрузка основных бэндов спутникового снимка
    for band_name in ["blue", "green", "red", "nir", "swir1", "swir2"]:
        if bands_path_list.get(band_name) is not None:
            if transform_matrix is None:
                band, (image_shape, transform_matrix, crs) = read_band(bands_path_list[band_name])
            else:
                band, _ = read_band(bands_path_list[band_name])
            bands_data[band_name] = resize(preprocess_band(band), image_shape)

    # Расчёт вег индексов
    for veg_index in bands_list:
        if veg_index in INDEX_BAND_MAPPER[level].keys():  # Проверка наличия необходимых каналов для индекса
            # logger.debug(f"level: {level}, veg_index: {veg_index}")
            formula = INDEX_FORMULAS.get(veg_index)
            if formula is None:
                raise ValueError(f"Formula for {veg_index} not found!")

            required_bands = INDEX_BANDS.get(veg_index, [])
            if len(required_bands) == 1:
                bands_data[veg_index] = formula(bands_data[required_bands[0]])
            elif len(required_bands) == 2:
                bands_data[veg_index] = formula(bands_data[required_bands[0]], bands_data[required_bands[1]])
            elif len(required_bands) == 3:
                bands_data[veg_index] = formula(
                    bands_data[required_bands[0]],
                    bands_data[required_bands[1]],
                    bands_data[required_bands[2]],
                )
            else:
                raise ValueError(f"Unsupported number of bands for {veg_index}: {len(required_bands)}")
        else:
            raise Exception(f"Band {veg_index} missing!")

    stacked_array = np.stack([bands_data[key] for key in bands_list if key in bands_data], axis=0)
    return np.transpose(stacked_array, (1, 2, 0)), (image_shape, transform_matrix, crs)


def min_max_normalize_with_clipping(image: np.ndarray, l_percent: int = 2, u_percent: int = 98) -> np.ndarray:
    """
    Min-max нормализация изображения с обрезкой крайних значений (по процентилям).

    image: Входное изображение (H, W, C).
    l_percent: Нижний процентиль для обрезки (по умолчанию 2%).
    u_percent: Верхний процентиль для обрезки (по умолчанию 98%).
    return: Нормализованное изображение (float32, значения от 0 до 1).
    """
    normalized_image = np.zeros_like(image, dtype=np.float32)  # Пустой массив для нормализованных значений

    for channel in range(image.shape[2]):  # Пробегаем по каждому каналу (R, G, B)
        channel_data = image[:, :, channel]

        # Вычисляем 2-й и 98-й процентили и нормализуем
        v_min, v_max = np.percentile(channel_data, [l_percent, u_percent])
        channel_data_clipped = np.clip(channel_data, v_min, v_max)
        normalized_image[:, :, channel] = (channel_data_clipped - v_min) / (
            v_max - v_min + 1e-6
        )  # 1e-6 для избежания деления на 0

    return normalized_image
