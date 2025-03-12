import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch.nn as nn
from loguru import logger
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
        brightened_rgb = enhance_rgb(output_img)

        if len(predict_mask.shape) == 3:
            predict_mask = (np.clip(predict_mask, 0, 1) > 0.5).astype(np.uint8)
            if predict_mask.shape[0] > 1:
                predict_mask = np.max(predict_mask[1:], axis=0)
        else:
            predict_mask = predict_mask.clip(0.3, 0.75)
        ground_truth_tensor = np.squeeze(ground_truth_tensor, axis=0).clip(0.3, 0.75)

        plt.figure(figsize=(30, 30))
        plt.subplot(2, 2, 1)
        plt.imshow(brightened_rgb)
        plt.title("Original RGB Image")
        plt.subplot(2, 2, 2)
        plt.imshow(brightened_rgb)
        plt.imshow(predict_mask, cmap="hot", alpha=0.5)
        plt.title("RGB Image + Model Mask")
        plt.subplot(2, 2, 3)
        plt.imshow(brightened_rgb)
        plt.imshow(ground_truth_tensor, cmap="hot", alpha=0.5)
        plt.title("RGB Image + Ground Truth Mask")
        plt.subplot(2, 2, 4)
        plt.imshow(ground_truth_tensor, cmap="gray")
        plt.imshow(predict_mask, cmap="hot", alpha=0.5)
        plt.title("Ground Truth Mask + Model Mask")
        plt.tight_layout()

        if file_name is None:
            save_path = f"{str(model_path)[:-4]}_{sample_num}.png"
            plt.savefig(save_path)
        else:
            save_path = evaluation_dir / f"{file_name}_{sample_num}.png"
            plt.savefig(save_path)

        logger.info(f"Evaluation result saved to: {save_path}")
        plt.close("all")

    return predict_mask


def enhance_rgb(image, lower_percent=2, upper_percent=98):
    """
    Улучшение контраста RGB-изображения через percentile stretching.

    image: np.array формы (H, W, 3)
    lower_percent, upper_percent: процентили для растяжения
    """
    enhanced_image = np.zeros_like(image, dtype=np.float32)
    for i in range(3):  # R, G, B
        channel = image[:, :, i]
        # Рассчитываем нижний и верхний процентили
        low, high = np.percentile(channel, (lower_percent, upper_percent))
        # Растягиваем динамический диапазон
        channel_stretched = np.clip((channel - low) / (high - low + 1e-6), 0, 1)
        enhanced_image[:, :, i] = channel_stretched

    # Дополнительно повысим яркость и насыщенность
    enhanced_image = np.clip(enhanced_image * 1.3, 0, 1)
    return enhanced_image


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
    logger.info(f"Average inference time: {avg_time:.6f} seconds")
