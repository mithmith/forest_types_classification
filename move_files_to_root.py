import os
import shutil
from pathlib import Path


def move_files_to_root(root_path: Path):
    """
    Перемещает все файлы из вложенных папок в корневую папку, заменяя существующие файлы, и удаляет пустые папки.
    
    :param root_path: Path объект, указывающий на корневую папку.
    """
    if not root_path.is_dir():
        print(f"Путь {root_path} не является директорией!")
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
            print(f"Перемещён файл {source} -> {destination}")

        # Удаляем пустую папку
        for dirname in dirnames:
            dir_to_remove = Path(dirpath) / dirname
            if not os.listdir(dir_to_remove):  # Проверяем, что папка пуста
                dir_to_remove.rmdir()
                print(f"Удалена пустая папка {dir_to_remove}")
    
    # Удаляем пустые папки в корне
    for dirpath, dirnames, _ in os.walk(root_path, topdown=False):
        for dirname in dirnames:
            dir_to_remove = Path(dirpath) / dirname
            if not os.listdir(dir_to_remove):
                dir_to_remove.rmdir()
                print(f"Удалена пустая папка {dir_to_remove}")


if __name__ == "__main__":
    root_folder = Path("./Разметка/8.Усыхание_Юг-Беларусь/S2B_MSIL2A_20210808T090559_N0301_R050_T35UPT_20210808T123030.SAFE/")
    move_files_to_root(root_folder)
