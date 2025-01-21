import numpy as np
from typing import List

def generate_ndvi_masks(ndvi: np.ndarray, n_mask: int = 3, min_delta: float = 0.1) -> List[np.ndarray]:
    """
    Разбивает NDVI массив на бинарные маски.

    Args:
        ndvi (np.ndarray): Массив с рассчитанными значениями NDVI.
        n_mask (int): Максимальное количество масок.
        min_delta (float): Минимальный шаг разбиения NDVI.

    Returns:
        list[np.ndarray]: Список бинарных масок.
    """
    # Игнорируем пустые значения (NaN)
    ndvi_valid = ndvi[~np.isnan(ndvi)]
    
    # Если нет валидных данных, возвращаем пустой список
    if ndvi_valid.size == 0:
        return []

    # Определяем минимальное и максимальное значения в массиве NDVI
    ndvi_min, ndvi_max = ndvi_valid.min(), ndvi_valid.max()

    # Если диапазон меньше минимального шага, возвращаем одну маску
    if (ndvi_max - ndvi_min) < min_delta:
        mask = ~np.isnan(ndvi)  # Все не NaN значения включаются
        return [mask]

    # Вычисляем фактический шаг разбиения
    step = (ndvi_max - ndvi_min) / n_mask
    if step < min_delta:
        step = min_delta
        n_mask = int((ndvi_max - ndvi_min) / step)

    # Генерируем пороговые значения
    thresholds = [ndvi_min + i * step for i in range(n_mask + 1)]

    # Создаём бинарные маски
    masks = []
    for i in range(n_mask):
        mask = (ndvi >= thresholds[i]) & (ndvi < thresholds[i + 1])
        mask[np.isnan(ndvi)] = False  # Сохраняем NaN на своих местах
        masks.append(mask)

    return masks

# Пример использования:
ndvi = np.random.uniform(-1, 1, (5, 5))  # Пример массива NDVI
ndvi[0, 0] = np.nan  # Добавляем NaN для тестирования
masks = generate_ndvi_masks(ndvi, n_mask=3, min_delta=0.1)

# Вывод результатов
print(ndvi)
for i, mask in enumerate(masks):
    print(f"Маска {i + 1}: ")
    print(mask)
