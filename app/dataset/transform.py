import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from scipy.ndimage import shift as shift_ndimage
from scipy.ndimage import zoom as zoom_ndimage


def random_shift(bands: list[np.ndarray], mask: np.ndarray, max_shift=50, p=0.25):
    """
    Случайный сдвиг (shift) всех каналов и маски вместе.
    :param bands: список numpy-массивов [H, W], по одному на каждый канал
    :param mask: numpy-массив [H, W], маска
    :param max_shift: максимальное смещение по каждой оси (x,y)
    :param p: вероятность применить трансформацию (0..1)
    :return: (shifted_bands, shifted_mask)
    """
    if np.random.rand() > p:
        # Без изменений
        return bands, mask

    # Генерируем случайные смещения по x и y
    shift_x = np.random.randint(-max_shift, max_shift + 1)
    shift_y = np.random.randint(-max_shift, max_shift + 1)

    shifted_bands = []
    for band in bands:
        # shift_ndimage сдвигает (top->bottom = ось 0, left->right = ось 1),
        # значит для y используем shift_y, для x — shift_x
        shifted_band = shift_ndimage(band, shift=(shift_y, shift_x), mode="constant", cval=0)
        shifted_bands.append(shifted_band)
    shifted_mask = shift_ndimage(mask, shift=(shift_y, shift_x), mode="constant", cval=0)

    return shifted_bands, shifted_mask


def random_zoom(bands: list[np.ndarray], mask: np.ndarray, scale_range=(0.8, 1.0), p=0.25):
    """
    Случайное уменьшение (zoom out), а затем обрезка/паддинг обратно к исходному размеру.
    :param bands: список numpy-массивов [H, W], по одному на каждый канал
    :param mask: numpy-массив [H, W], маска
    :param scale_range: (min_scale, max_scale) – диапазон случайного масштаба
    :param p: вероятность применить трансформацию (0..1)
    :return: (zoomed_bands, zoomed_mask)
    """
    if np.random.rand() > p:
        return bands, mask

    # Исходные размеры
    h, w = mask.shape
    # Случайный фактор уменьшения
    scale = np.random.uniform(scale_range[0], scale_range[1])
    # Если scale = 1.0, трансформации не будет

    zoomed_bands = []
    for band in bands:
        # zoom_ndimage меняет размер, фактор zoom задаётся для каждой оси
        # В нашем случае изображение [H, W], значит scale_factors = (scale, scale)
        band_zoomed = zoom_ndimage(band, zoom=scale, order=1)  # bilinear
        zoomed_bands.append(band_zoomed)

    mask_zoomed = zoom_ndimage(mask, zoom=scale, order=0)  # nearest для маски

    # Теперь нам нужно вернуть исходный размер [h, w].
    # Либо мы берём центральный кроп, если уменьшили,
    # либо делаем паддинг, если scale<1 (хотя scale<1 => картинка меньше).
    def center_crop_or_pad(image, out_h, out_w):
        in_h, in_w = image.shape
        # Если in_h > out_h – нужно обрезать, если меньше – паддинг
        start_y = max(0, (in_h - out_h) // 2)
        start_x = max(0, (in_w - out_w) // 2)

        end_y = start_y + out_h
        end_x = start_x + out_w

        cropped = image
        if in_h > out_h or in_w > out_w:
            # обрежем по центру
            cropped = cropped[start_y:end_y, start_x:end_x]

        # Если после обрезки картинка меньше требуемого размера – сделаем паддинг
        # (по центру)
        final = np.zeros((out_h, out_w), dtype=cropped.dtype)
        final_h, final_w = cropped.shape
        pad_y = (out_h - final_h) // 2
        pad_x = (out_w - final_w) // 2
        final[pad_y : pad_y + final_h, pad_x : pad_x + final_w] = cropped

        return final

    zoomed_bands = [center_crop_or_pad(band_zoomed, h, w) for band_zoomed in zoomed_bands]
    zoomed_mask = center_crop_or_pad(mask_zoomed, h, w)

    return zoomed_bands, zoomed_mask


def random_blur_or_sharpen(bands: list[np.ndarray], mask: np.ndarray, blur_prob=0.5, sigma_range=(0.5, 2.0), sharpen_amount=1.0, p=0.2):
    """
    Случайное применение Gaussian blur или усиления резкости (unsharp mask).
    ВАЖНО: маску (mask) обычно НЕ размывают и не шарпят, т.к. это ломает разметку.
    :param bands: список numpy-массивов [H, W], по одному на каждый канал (например, float в диапазоне [0..1])
    :param mask: numpy-массив [H, W], маска сегментации
    :param blur_prob: вероятность выбрать именно blur (иначе sharpen)
    :param sigma_range: диапазон значений sigma для гаусс. размытия
    :param sharpen_amount: коэффициент усиления резкости
    :param p: вероятность вообще применить (0..1)
    :return: (new_bands, mask) – маска не меняется!
    """
    if np.random.rand() > p:
        return bands, mask

    # Выбираем, что делать: blur или sharpen
    do_blur = np.random.rand() < blur_prob
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])

    new_bands = []
    for band in bands:
        if do_blur:
            # Gaussian blur
            blurred = gaussian_filter(band, sigma=sigma)
            new_bands.append(blurred)
        else:
            # Unsharp mask: резкость = band + amount*(band - blurred)
            blurred = gaussian_filter(band, sigma=sigma)
            sharpened = band + sharpen_amount * (band - blurred)
            # ограничим значение в допустимых пределах
            sharpened = np.clip(sharpened, 0, 1)
            new_bands.append(sharpened)

    # Маску не трогаем
    return new_bands, mask


def random_haze(band: np.ndarray, alpha_range=(0.2, 0.8), p=0.2):
    """
    Добавление «атмосферного» эффекта дымки (haze). Маска не меняется.
    Для упрощения считаем, что пиксели band в [0..1].
    :param band: numpy-массив [H, W]
    :param mask: numpy-массив [H, W]
    :param alpha_range: (min_alpha, max_alpha) – коэффициент прозрачности
    :param p: вероятность применить трансформацию
    :return: (new_bands, mask)
    """
    if np.random.rand() > p:
        return band

    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    # Возьмём "белый" как 1.0 (если у нас нормировка [0..1])
    # Формула haze: new_pixel = alpha * pixel + (1 - alpha) * 1.0

    haze_band = alpha * band + (1 - alpha) * 1.0
    return np.clip(haze_band, 0, 1)


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


def add_gaussian_noise(image: np.ndarray, std_dev: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0, std_dev, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)


def add_random_rotation_and_flip(bands: list[np.ndarray], mask: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
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
        bands = [rotate(band, angle, axes=(0, 1), reshape=False) for band in bands]
        mask = rotate(mask, angle, axes=(0, 1), reshape=False)

    # Случайное отражение
    if np.random.rand() > 0.5:  # Отражение по ширине
        bands = [np.flip(band, axis=1) for band in bands]
        mask = np.flip(mask, axis=1)

    if np.random.rand() > 0.5:  # Отражение по высоте
        bands = [np.flip(band, axis=0) for band in bands]
        mask = np.flip(mask, axis=0)

    return bands, mask
