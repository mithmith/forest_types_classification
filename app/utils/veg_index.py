import cv2
import numpy as np


def ndvi_formula(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.clip((nir - red) / (nir + red), -1, 1)


def evi_formula(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        znam = nir + 6 * red - 7.5 * blue + 1
        return np.clip(2.5 * (nir - red) / znam, -1, 1)


def preprocess_band(band: np.ndarray) -> np.ndarray:
    return np.clip(np.array(band / 10000 - 0.1, dtype=np.float32), 0, 1)


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
