import json
from datetime import datetime
from functools import partial
from typing import Any

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from loguru import logger
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shapely_transform


def mask_to_polygons(mask: np.ndarray) -> list[Polygon]:
    """
    Преобразует маску в массив полигонов.

    Args:
        mask (np.ndarray): Маска предсказаний.

    Returns:
        list[Polygon]: Массив найденных полигонов.
    """
    mask = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons: list[Polygon] = []
    for contour in contours:
        contour = contour.squeeze(1)
        if contour.shape[0] >= 4:  # в контуре более 4 точек
            poly = Polygon(contour)
            polygons.append(poly)
    return polygons


def mask_from_geojson(geojson_data: dict, mask_shape: tuple[int, int], transform_matrix: Affine, crs) -> np.ndarray:
    """
    Преобразует GeoJSON с полигонами в объединённую бинарную маску.

    Args:
        geojson_data (dict): Данные GeoJSON, содержащие полигоны.
        mask_shape (tuple[int, int]): Размер маски в формате (высота, ширина).
        transform_matrix (Affine): Аффинная матрица трансформации для привязки геометрии к растровым данным.

    Returns:
        np.ndarray: Объединённая бинарная маска.
    """
    # Преобразуем GeoJSON в GeoDataFrame
    geo_df = gpd.GeoDataFrame.from_features(geojson_data["features"])

    geo_df_crs = geojson_data["crs"]["properties"]["name"].split(":")[-1]

    if geo_df_crs == "CRS84":
        geo_df = geo_df.set_crs("EPSG:4326")
        geo_df = geo_df.to_crs(crs)

    geometries = [(shape(row["geometry"]), 1) for _, row in geo_df.iterrows() if row["geometry"] is not None]

    combine_mask = rasterize(
        geometries, out_shape=mask_shape, transform=transform_matrix, fill=0, all_touched=True, dtype=np.uint8
    )

    return combine_mask


def pixel_to_geo_coords(polygon: Polygon, transform_matrix: Affine) -> Polygon:
    """
    Преобразует координаты полигона из пиксельных координат в географические.

    Args:
        polygon (Polygon): Полигон с пиксельными координатами.
        transform_matrix (Affine): Матрица трансформации, использованная для перевода географических координат в пиксели.

    Returns:
        Polygon: Полигон с географическими координатами.
    """

    # Функция для преобразования одной точки с использованием аффинной матрицы
    def transform_point(affine_transform, x, y, z=None):
        gx, gy = affine_transform * (x, y)
        return gx, gy

    # Создаем partial функцию с заданным affine_transform
    transform_func = partial(transform_point, transform_matrix)

    # Применяем transform_func ко всем точкам полигона
    transformed_polygon = shapely_transform(lambda x, y, z=None: transform_func(x, y, z), polygon)

    return transformed_polygon


def polygon_to_pixel_coords(polygon: Polygon, transform_matrix: Affine) -> Polygon:
    return pixel_to_geo_coords(polygon, ~transform_matrix)


def create_geojson(polygons: list[Polygon], crs: rasterio.crs.CRS) -> dict[str, Any]:
    """Создает GeoJSON из списка полигонов."""
    features = []
    for idx, poly in enumerate(polygons):
        # Создание feature для каждого полигона
        feature = {
            "type": "Feature",
            "geometry": mapping(poly),
            "properties": {
                "id": idx,
                "create_at": datetime.now().isoformat(),  # Текущая дата и время в ISO формате
                "model_key": "fieldworks",
                "area": poly.area,  # Площадь полигона
            },
        }
        features.append(feature)

    # Преобразуем CRS в формат EPSG (например, "EPSG:4326")
    crs_epsg = crs.to_string()

    # Создание FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {"type": "name", "properties": {"name": crs_epsg}},
    }
    return geojson


def raster_to_geojson(prediction_image: np.ndarray, transform_matrix: Affine, crs, output_geojson: str) -> dict:
    """Преобразует растровую маску предсказаний в GeoJSON."""
    # Преобразование маски в бинарный формат
    mask = (prediction_image > 0.5).astype("uint8") * 255
    polygons = mask_to_polygons(mask)
    logger.debug(f"polygons count in mask: {len(polygons)}")

    polygons = [pixel_to_geo_coords(poly, transform_matrix) for poly in polygons]
    polygons = [transform_polygon_to_crs(poly, crs.to_string(), crs.to_string()) for poly in polygons]
    geojson = create_geojson(polygons, crs)

    # Сохранение GeoJSON в файл
    with open(output_geojson, "w", encoding="utf8") as f:
        json.dump(geojson, f)
    return geojson


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


def transform_polygon_to_crs(polygon: Polygon | MultiPolygon, src_crs: str, dst_crs: str) -> Polygon | MultiPolygon:
    """
    Преобразует координаты полигона из одной системы координат в другую.

    Args:
        polygon (Polygon | MultiPolygon): Полигон или мультиполигон с координатами в исходной системе координат.
        src_crs (str): Исходная система координат (например, 'EPSG:4326').
        dst_crs (str): Целевая система координат (например, 'EPSG:3857').

    Returns:
        Polygon | MultiPolygon: Полигон или мультиполигон с координатами в целевой системе координат.
    """
    # Создаем трансформер для преобразования из исходной CRS в целевую CRS
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # Функция для трансформации координат
    def transform_point(x, y):
        return transformer.transform(x, y)

    # Применяем трансформацию к полигону или мультиполигону
    transformed_polygon = shapely_transform(transform_point, polygon)

    return transformed_polygon


def polygon_to_mask(polygon: Polygon | MultiPolygon, image_shape: tuple[int, int], transform_matrix) -> np.ndarray:
    """
    Преобразует полигон или мультиполигон в бинарную маску.

    Args:
        polygon (Polygon | MultiPolygon): Полигон или мультиполигон, который нужно преобразовать в маску.
        image_shape (tuple[int, int]): Размер изображения в формате (высота, ширина).
        transform_matrix (Affine): Аффинная матрица трансформации для привязки геометрии к растровым данным.

    Returns:
        np.ndarray: Бинарная маска, где значения 1 соответствуют полигону/мультиполигону, а 0 — фону.
    """
    # Инициализируем пустую маску
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Преобразуем полигон в формат геометрии для rasterio
    geometries = [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)

    # Растеризуем полигон на маску
    mask = rasterize(
        [(geom, 1) for geom in geometries],  # Присваиваем значение 1 там, где есть полигон
        out_shape=image_shape,
        transform=transform_matrix,  # Используем матрицу трансформации
        fill=0,  # По умолчанию заполняем нулями
        all_touched=True,  # Это обеспечивает более полное покрытие краёв полигона
        dtype=np.uint8,  # Маска будет содержать значения 0 и 1
    )

    return mask


def feature_to_polygon(feature) -> Polygon | MultiPolygon:
    """
    Преобразует геометрию объекта Feature в объект Polygon или MultiPolygon.

    Функция принимает объект Feature, который содержит геометрию в формате GeoJSON,
    и преобразует её в объект Polygon или MultiPolygon из библиотеки Shapely.

    - Для типа "Polygon", первый элемент координат представляет внешний контур,
      остальные (если присутствуют) — внутренние контуры (дырки).
    - Для типа "MultiPolygon", каждый элемент представляет отдельный полигон,
      состоящий из внешнего контура и, при наличии, внутренних контуров.

    Args:
        feature (Feature): Объект Feature, содержащий геометрию в формате GeoJSON.
                           Предполагается, что геометрия имеет тип "Polygon" или "MultiPolygon".

    Returns:
        Polygon | MultiPolygon: Объект типа Polygon или MultiPolygon,
                                в зависимости от типа геометрии входного объекта.
    """
    field_geom = None

    # Получаем геометрию поля
    if feature.geometry.type == "Polygon":
        # Первый элемент - внешний контур, остальные (если есть) - внутренние контуры (дырки)
        exterior = feature.geometry.coordinates[0]
        interiors = feature.geometry.coordinates[1:] if len(feature.geometry.coordinates) > 1 else []
        field_geom = Polygon(exterior, interiors)

    elif feature.geometry.type == "MultiPolygon":
        polygons: list[Polygon] = []
        for poly_coords in feature.geometry.coordinates:
            exterior = poly_coords[0]  # Внешний контур
            interiors = poly_coords[1:] if len(poly_coords) > 1 else []  # Дырки
            polygons.append(Polygon(exterior, interiors))
        field_geom = MultiPolygon(polygons)

    return field_geom
