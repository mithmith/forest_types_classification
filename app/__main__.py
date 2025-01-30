import os
import random
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from clearml import Task

from app.dataset_single import ForestTypesDataset
from app.utils.veg_index import preprocess_band

# os.environ["GDAL_DATA"] = os.environ["CONDA_PREFIX"] + r"\Library\share\gdal"
# os.environ["PROJ_LIB"] = os.environ["CONDA_PREFIX"] + r"\Library\share"
# os.environ["GDAL_DRIVER_PATH"] = os.environ["CONDA_PREFIX"] + r"\Library\lib\gdalplugins"

# # Specify the path
# path = Path("G:/Orni_forest/forest_changes_dataset/generated_dataset")

# dataset_geojson_masks_dir = Path("G:/Orni_forest/forest_changes_dataset/masks")
# sentinel_root_dir = Path("G:/Orni_forest/forest_changes_dataset/images")
# train_dataset = ForestTypesDataset(
#     dataset_geojson_masks_dir,
#     sentinel_root_dir,
#     dataset_path=Path("G:/Orni_forest/forest_changes_dataset/generated_dataset/train"),
# )
# val_dataset = ForestTypesDataset(
#     dataset_geojson_masks_dir,
#     sentinel_root_dir,
#     dataset_path=Path("G:/Orni_forest/forest_changes_dataset/generated_dataset/validation"),
# )

# 1. Generate dataset samples
# train_dataset.generate_dataset(target_samples=4500)
# exit()
# 2. Get generated samples for learning
# sample, mask = next(train_dataset.get_next_generated_sample(exclude_nir=False))
# print(sample.shape, mask.shape)
# print("Sample min/max/avr:\t", np.nanmin(sample), np.nanmax(sample), np.nanmean(sample))
# print("Mask min/max/avr:\t", np.nanmin(mask), np.nanmax(mask), np.nanmean(mask))


# # 3. ВИЗУАЛИЗАЦИЯ СТРУКТУРЫ МОДЕЛИ:
# # Save model summary structure:
# from torchsummary import summary
# from torchviz import make_dot
# from app.modelResNet50_RGB_NIR import ResNet50_RGB_NIR_Model
# from app.modelSKResNeXt50_UNet import SKResNeXt50_UNet

# model = SKResNeXt50_UNet(num_classes=1)
# rnd_sample = torch.randn(1, 3, 512, 512)
# summary(model, (3, 512, 512), batch_size=1, device="cpu")
# output = model(rnd_sample)
# dot = make_dot(output, params=dict(model.named_parameters()))
# # Save or display the generated graph
# dot.format = "png"
# dot.render("SKResNeXt50_UNet")

# from torchview import draw_graph

# # Создание графа модели
# model_graph = draw_graph(model, input_size=[(1, 3, 512, 512)], expand_nested=True)
# # Сохранение графа в PNG файл
# model_graph.visual_graph.render(filename="SKResNeXt50_UNet model_structure", format="png", cleanup=True)
# exit()

# Загружаем модель
# model.load_model(f"G:/Orni_forest/sentinel_forest_types_classification/drying_classic_unet_models_3masks/forest_segmentation_resnet_v{i}.pth")
# model.logs_path = Path(f"./ResNet50_RGB_Model_drying/train_progress_v{1}_drying/")
# task = Task.init(project_name='T1-ML-ResNet50_RGB_Model-drying', task_name=f'Forest_changes_v{1}', reuse_last_task_id=False)
# # Обучаем модель на датасете
# model.train_model(train_dataset=train_dataset, val_dataset=val_dataset, epochs=10, batch_size=1, learning_rate=0.001)
# # Сохраняем обученную модель в файл
# model.save_model(f"G:/Orni_forest/sentinel_forest_types_classification/ResNet50_RGB_Model_drying/forest_segmentation_resnet_v{1}.pth")
#
# task.close()

# model = ResNet50_RGB_NIR_Model(num_classes=1)
#
# # Загружаем модель
# model.logs_path = Path(f"./ResNet50_RGB_NIR_Model_drying/train_progress_v{1}_drying/")
# task = Task.init(project_name='T1-ML-ResNet50_RGB_NIR_Model-drying', task_name=f'Forest_changes_v{1}', reuse_last_task_id=False)
# # Обучаем модель на датасете
# model.train_model(train_dataset=train_dataset, val_dataset=val_dataset, epochs=10, batch_size=1, learning_rate=0.001)
# # Сохраняем обученную модель в файл
# model.save_model(f"G:/Orni_forest/sentinel_forest_types_classification/ResNet50_RGB_NIR_Model_drying/forest_segmentation_resnet_v{1}.pth")
#
# task.close()


# Проверка работы модели
def predict_sample_from_dataset(dataset_dir: Path, sample_num: int):
    features_names = ["red", "green", "blue", "nir"]
    input_tensor = []
    for feature_name in features_names:
        with rasterio.open(dataset_dir / f"{sample_num}_{feature_name}.tif") as f:
            input_tensor.append(preprocess_band(f.read(1)))

    # input_tensor.append(
    #     train_dataset.create_forest_mask(
    #         nir_path=dataset_dir / f"{sample_num}_nir.tif",
    #         red_path=dataset_dir / f"{sample_num}_red.tif",
    #         blue_path=dataset_dir / f"{sample_num}_blue.tif",
    #     )
    # )

    input_tensor = np.array(input_tensor)
    model = ResNet50_RGB_NIR_Model(num_classes=1)
    model.load_model(
        f"G:/Orni_forest/sentinel_forest_types_classification/ResNet50_RGB_NIR_Model_drying/forest_segmentation_resnet_v{1}.pth"
    )

    num_runs = 100
    times = []

    for _ in range(num_runs):
        start_time = time.perf_counter()

        predict_mask = model.evaluate(input_tensor)

        end_time = time.perf_counter()
        times.append(end_time - start_time)

    avg_time = sum(times) / num_runs
    print(f"Average inference time: {avg_time:.6f} seconds")

    # Визуализируем результаты
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(input_tensor[0], cmap="gray")
    # plt.subplot(1, 3, 2)
    # plt.imshow(predict_mask.clip(0.3, 0.75), cmap="gray")
    # plt.show()


dataset_dir = Path("G:/Orni_forest/forest_changes_dataset/generated_dataset/train")
red_tif_files = list(dataset_dir.glob("*_red.tif"))
random.shuffle(red_tif_files)
for filename in red_tif_files[:5]:
    n = filename.stem.split("_")[0]
    val_dataset.inference_test(
        model, model_save_path.joinpath(f"{model_name}_v{i}.pth"), n, 100, exclude_nir=True, exclude_fMASK=True
    )

task.close()
