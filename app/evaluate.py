import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch.nn as nn
from osgeo import gdal

from app.dataset_single import ForestTypesDataset
from app.train import evaluate, load_model
from app.utils import veg_index


def predict_sample_from_dataset(
    model: nn.Module,
    model_path: Path,
    sample_num: str,
    dataset_path: Path,
    forest_model_path: Path,
    exclude_nir=False,
    exclude_fMASK=False,
    evaluation_dir=None,
    file_name=None,
):
    features_names = ["red", "green", "blue"]

    if not exclude_nir:
        features_names.append("nir")

    input_tensor = []
    output_img = []
    for feature_name in features_names:
        with rasterio.open(dataset_path / f"{sample_num}_{feature_name}.tif") as f:
            if feature_name != "nir":
                in_ds = gdal.OpenEx(str(dataset_path / f"{sample_num}_{feature_name}.tif"))
                out_ds = gdal.Translate("/vsimem/in_memory_output.tif", in_ds)
                out_arr = out_ds.ReadAsArray()
                output_img.append(out_arr)
            input_tensor.append(veg_index.preprocess_band(f.read(1)))

    ground_truth_tensor = []
    with rasterio.open(dataset_path / f"{sample_num}_mask.tif") as f:
        ground_truth_tensor.append(f.read(1))

    if not exclude_fMASK:
        input_tensor.append(ForestTypesDataset.create_forest_mask(sample_num, dataset_path, forest_model_path))

    input_tensor = np.array(input_tensor)
    loaded_model = load_model(model, model_path)

    predict_mask = evaluate(loaded_model, input_tensor)

    if evaluation_dir is not None and evaluation_dir.exists():
        output_img = np.transpose(output_img, (1, 2, 0))
        normalized_rgb = np.zeros_like(output_img, dtype=np.float32)  # Создаём пустой массив для нормализации
        for channel in range(output_img.shape[2]):  # По каждому каналу (R, G, B)
            channel_data = output_img[:, :, channel]
            normalized_rgb[:, :, channel] = (channel_data - channel_data.min()) / (
                channel_data.max() - channel_data.min() + 1e-6
            )  # Добавляем 1e-6 для избежания деления на 0
        # Increase brightness by scaling up values (factor 1.5 can be adjusted)
        brightness_factor = 3
        brightened_rgb = np.clip(normalized_rgb * brightness_factor, 0, 1)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(brightened_rgb)
        plt.imshow(predict_mask.clip(0.3, 0.75), cmap="hot", alpha=0.5)
        plt.title("RGB Image + Model Mask")
        plt.subplot(1, 3, 2)
        plt.imshow(brightened_rgb)
        plt.imshow(np.squeeze(ground_truth_tensor, axis=0).clip(0.3, 0.75), cmap="hot", alpha=0.5)
        plt.title("RGB Image + Ground Truth Mask")
        plt.subplot(1, 3, 3)
        ground_truth = np.squeeze(ground_truth_tensor, axis=0).clip(0.3, 0.75)
        plt.imshow(ground_truth, cmap="gray")
        plt.imshow(predict_mask.clip(0.3, 0.75), cmap="hot", alpha=0.5)
        plt.title("Ground Truth Mask + Model Mask")
        plt.tight_layout()

        if file_name is None:
            save_path = f"{str(model_path)[:-4]}_{sample_num}.png"
            plt.savefig(save_path)
        else:
            save_path = evaluation_dir / f"{file_name}_{sample_num}.png"
            plt.savefig(save_path)

        print(f"Evaluation result saved to: {save_path}")
        plt.close("all")

    return predict_mask


def inference_test(
    model: nn.Module,
    model_path: Path,
    sample_num: str,
    num_runs: int,
    dataset_path: Path,
    forest_model_path: Path,
    exclude_nir=False,
    exclude_fMASK=False,
):
    features_names = ["red", "green", "blue"]

    if not exclude_nir:
        features_names.append("nir")

    input_tensor = []
    for feature_name in features_names:
        with rasterio.open(dataset_path / f"{sample_num}_{feature_name}.tif") as f:
            input_tensor.append(veg_index.preprocess_band(f.read(1)))

    ground_truth_tensor = []
    with rasterio.open(dataset_path / f"{sample_num}_mask.tif") as f:
        ground_truth_tensor.append(f.read(1))

    if not exclude_fMASK:
        input_tensor.append(ForestTypesDataset.create_forest_mask(sample_num, dataset_path, forest_model_path))

    input_tensor = np.array(input_tensor)
    loaded_model = load_model(model, model_path)

    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()

        predict_mask = evaluate(loaded_model, input_tensor)

        end_time = time.perf_counter()
        times.append(end_time - start_time)

    avg_time = sum(times) / num_runs
    print(f"Average inference time: {avg_time:.6f} seconds")
