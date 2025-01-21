import os
import random
from os import listdir
from os.path import isfile, join
from typing import Any, Dict, List

import boto3
import pandas as pd
import requests


def get_access_token(username: str, password: str) -> str:
    """Получает токен доступа для аутентификации в системе."""
    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    response = requests.post(
        auth_url,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    response.raise_for_status()
    access_token = response.json().get("access_token")

    if not access_token:
        raise Exception("Не удалось получить access_token")

    return access_token


def extract_image_names(file_path: str) -> List[str]:
    """
    Извлекает названия изображений из Excel-файла.

    Args:
        file_path (str): Путь к Excel-файлу.

    Returns:
        List[str]: Список названий продуктов.
    """
    # Проверка существования файла
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    xls = pd.ExcelFile(file_path)

    all_image_names = []
    for sheet_name in xls.sheet_names:
        # Извлечение названий из столбца G (индекс 6), начиная с 3-й строки, во всех листах
        df_before = pd.read_excel(xls, sheet_name=sheet_name, usecols=[6], skiprows=2)
        # Предполагается, что названия находятся в первом столбце после загрузки
        image_names = df_before.iloc[:, 0].dropna().astype(str).tolist()
        all_image_names.extend(image_names)

        # Извлечение названий из столбца J (индекс 9), начиная с 3-й строки, во всех листах
        df_after = pd.read_excel(xls, sheet_name=sheet_name, usecols=[9], skiprows=2)
        # Предполагается, что названия находятся в первом столбце после загрузки
        image_names = df_after.iloc[:, 0].dropna().astype(str).tolist()
        all_image_names.extend(image_names)

    unique_image_names = list(set(all_image_names))

    # Добавление окончания .SAFE к каждому названию, если его нет
    unique_image_names_with_safe = [f"{name}.SAFE" for name in unique_image_names]

    return unique_image_names_with_safe


def build_odata_url(product_names: List[str], processing_levels: List[str]) -> str:
    """Строит URL для запроса продуктов по названиям и уровням обработки."""
    # Экранирование одинарных кавычек в названиях продуктов
    escaped_names = [name.replace("'", "''") for name in product_names]

    # Фильтр по названиям продуктов
    name_filters = " or ".join([f"Name eq '{name}'" for name in escaped_names])

    # Фильтр по уровням обработки
    level_filters = " or ".join(
        [
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{level}')"
            for level in processing_levels
        ]
    )

    # Полный фильтр
    full_filter = f"({name_filters}) and ({level_filters})"

    # URL-энкодинг фильтра
    from urllib.parse import quote

    encoded_filter = quote(full_filter)

    return f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?" f"$filter={encoded_filter}"


def get_products(session: requests.Session, odata_url: str) -> List[Dict[str, Any]]:
    """Получает список продуктов из API."""
    response = session.get(odata_url)
    if response.status_code == 204:
        # Нет содержимого
        return []
    response.raise_for_status()
    return response.json().get("value", [])


def download_files_from_s3(bucket, product: str, target: str = "Sentinel-2") -> None:
    """Скачивает все файлы из S3 бакета с указанным префиксом."""
    # Удаляем начальный префикс '/eodata/' из пути, если он есть
    product = product.lstrip("/eodata/")

    # Извлекаем имя снимка (последний элемент пути)
    product_name = os.path.basename(product)

    # Создаём целевую папку с именем снимка
    target_folder = os.path.join(target, product_name)
    os.makedirs(target_folder, exist_ok=True)

    files = list(bucket.objects.filter(Prefix=product))

    if not files:
        print(f"Предупреждение: Не удалось найти файлы для {product}")
        return

    for file in files:
        # Создание пути для файла в целевой папке
        local_file_path = os.path.join(target_folder, os.path.basename(file.key))

        # Проверка, загружен ли файл уже
        if os.path.exists(local_file_path):
            print(f"Файл уже скачан: {local_file_path}")
            continue

        # Скачивание файла
        print(f"Скачивание файла: {local_file_path}")
        try:
            bucket.download_file(file.key, local_file_path)
        except Exception as e:
            print(f"Ошибка при скачивании {file.key}: {e}")


def main(
    username: str,
    password: str,
    excel_file_path: str,
    processing_levels: List[str],
    access_key: str,
    secret_key: str,
    download: bool = True,
    target_dir: str = "Sentinel-2",
) -> None:
    """Основная функция для получения и скачивания продуктов Sentinel по названиям из Excel."""

    product_names = extract_image_names(excel_file_path)

    mypath = "E:/satellite/Sentinel-2/"
    downloaded_folders = [f for f in listdir(mypath)]
    product_names = list(set(product_names) - set(downloaded_folders))
    print(f"Найдено {len(product_names)} уникальных названий.\n")

    access_token = get_access_token(username, password)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {access_token}"})

    downloaded_images = []
    list_len = 0
    while len(product_names) > 0:
        k = 20
        if k > len(product_names):
            k = len(product_names)
        odata_url = build_odata_url(random.sample(product_names, k), processing_levels)

        print(f"{len(product_names)} снимков осталось скачать")
        print(f"Запрос OData URL: {odata_url}")

        products = get_products(session, odata_url)

        print(f"По запросу найдено {len(products)} файлов!\n")

        if download and products:
            s3_session = boto3.session.Session()
            s3 = s3_session.resource(
                service_name="s3",
                endpoint_url="https://eodata.dataspace.copernicus.eu",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="default",
            )
            bucket = s3.Bucket("eodata")

            for product in products:
                s3_path = product.get("S3Path")
                if s3_path:
                    download_files_from_s3(bucket, s3_path, target=target_dir)

            for product in products:
                downloaded_images.append(product.get("Name", "Unnamed Product"))
            product_names = list(set(product_names) - set(downloaded_images))
            if list_len != len(product_names):
                list_len = len(product_names)
            else:
                print(f"Не удалось скачать {len(product_names)} снимков:")
                for product in product_names:
                    print(product)
                break
        elif not download:
            for product in products:
                downloaded_images.append(product.get("Name", "Unnamed Product"))
            product_names = list(set(product_names) - set(downloaded_images))
            if list_len != len(product_names):
                list_len = len(product_names)
            else:
                print(f"Не удалось скачать {len(product_names)} снимков:")
                for product in product_names:
                    print(product)
                break


if __name__ == "__main__":
    import time

    import config

    # Указываем параметры для поиска
    username = config.username or "E-MAIL"
    password = config.password or "PASSWORD"
    access_key = config.access_key or "YOUR_ACCESS_KEY"
    secret_key = config.secret_key or "YOUR_SECRET_KEY"

    # Путь к Excel-файлу с названиями продуктов
    excel_file_path = "./image_names.xlsx"

    # Уровни обработки продуктов
    processing_levels = ["L1C", "L2A"]  # ['L1C', 'L2A']

    while True:
        try:
            # Запуск основного процесса
            main(
                username=username,
                password=password,
                excel_file_path=excel_file_path,
                processing_levels=processing_levels,
                access_key=access_key,
                secret_key=secret_key,
                download=True,
                target_dir="E:/satellite/Sentinel-2/",
            )
            break
        except Exception as e:
            print("Function errored out!", e)
            print("Retrying ... ")
            time.sleep(5)
