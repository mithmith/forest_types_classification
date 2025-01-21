from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from clearml import Task

from app.dataset import ForestTypesDataset
from app.model import DualEncoderUNetModel

dataset_geojson_masks_dir = Path("./")
sentinel_root_dir = Path("E:/satellite/Sentinel-2/")
dataset = ForestTypesDataset(dataset_geojson_masks_dir, sentinel_root_dir, dataset_path=Path("./generated_dataset/"))
# 1. Generate dataset samples
dataset.generate_dataset(num_samples=25)
exit()
# 2. Get generated samples for learning
sample1, sample2, mask = next(dataset.get_next_generated_sample())
print(sample1.shape, sample2.shape, mask.shape)
print("Sample1 min/max/avr:\t", np.nanmin(sample1), np.nanmax(sample1), np.nanmean(sample1))
print("Sample2 min/max/avr:\t", np.nanmin(sample2), np.nanmax(sample2), np.nanmean(sample2))
print("Mask min/max/avr:\t", np.nanmin(mask), np.nanmax(mask), np.nanmean(mask))

model = DualEncoderUNetModel(num_classes=1)

# Save model summary structure:
# from torchsummary import summary
# from torchviz import make_dot
# summary(model, [(5, 512, 512), (5, 512, 512)], batch_size=1, device="cpu")
# output = model(model.prepare_input(sample1), model.prepare_input(sample2))
# dot = make_dot(output, params=dict(model.named_parameters()))
# # Save or display the generated graph
# dot.format = 'png'
# dot.render('DualEncoderUNetModel_net')
# from torchview import draw_graph
# model_graph = draw_graph(model, input_size=[(1,5,512,512), (1,5,512,512)], expand_nested=True)
# model_graph.visual_graph
# exit()

# # Загружаем модель
# model.load_model("forest_segmentation_resnet_v82_unfreezed.pth")
# model.logs_path = Path("./train_progress_v8_3/")
# task = Task.init(project_name='T1-ML', task_name='Forest_changes_v83')
# # Обучаем модель на датасете
# model.train_model(dataset, epochs=1, batch_size=1, learning_rate=0.001, freeze_rgb=False)
# # Сохраняем обученную модель в файл
# model.save_model("forest_segmentation_resnet_v83_unfreezed.pth")

# Проверка работы модели
import rasterio
from app.utils.veg_index import preprocess_band

def predict_sample_from_dataset(dataset_dir: Path, sample_num: int):
    features_names = ["red", "green", "blue", "nir"]    # RGB + nir
    input_tensor1, input_tensor2 = [], []
    for feature_name in features_names:
        with rasterio.open(dataset_dir / f"{sample_num}_{feature_name}_1.tif") as f:
            input_tensor1.append(preprocess_band(f.read(1)))
    for feature_name in features_names:
        with rasterio.open(dataset_dir / f"{sample_num}_{feature_name}_2.tif") as f:
            input_tensor2.append(preprocess_band(f.read(1)))
    input_tensor1.append(
        dataset.create_forest_mask(
            nir_path=dataset_dir / f"{sample_num}_nir_1.tif",
            red_path=dataset_dir / f"{sample_num}_red_1.tif",
            blue_path=dataset_dir / f"{sample_num}_blue_1.tif",
        )
    )
    input_tensor2.append(
        dataset.create_forest_mask(
            nir_path=dataset_dir / f"{sample_num}_nir_2.tif",
            red_path=dataset_dir / f"{sample_num}_red_2.tif",
            blue_path=dataset_dir / f"{sample_num}_blue_2.tif",
        )
    )
    input_tensor1 = np.array(input_tensor1)
    input_tensor2 = np.array(input_tensor2)
    model.load_model("forest_segmentation_resnet_v86_unfreezed.pth")
    predict_mask = model.evaluate(input_tensor1, input_tensor2)
    print(input_tensor1.shape)
    print(input_tensor2.shape)
    print(predict_mask.shape)

    # Визуализируем результаты
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(input_tensor1[0], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(input_tensor2[0], cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(predict_mask.clip(0.3, 0.75), cmap="gray")
    plt.show()

import random
dataset_dir = Path("D:/usr/T1-GIS/sentinel_forest_types_classification/generated_dataset/")
sample_num = 100025
predict_sample_from_dataset(dataset_dir, sample_num)

# red_tif_files = list(dataset_dir.glob("*_red_*.tif"))
# random.shuffle(red_tif_files)
# for filename in red_tif_files:
#     # Извлекаем номер выборки и путь до маски
#     n = int(filename.stem.split("_")[0])
#     predict_sample_from_dataset(dataset_dir, n)
