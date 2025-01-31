import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch.nn as nn

from app.dataset_single import ForestTypesDataset
from app.train import evaluate, load_model
from app.utils import veg_index


def predict_sample_from_dataset(
    model: nn.Module,
    model_path: Path,
    sample_num: str,
    dataset_path: Path,
    exclude_nir=False,
    exclude_fMASK=False,
    visualize=False,
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
        input_tensor.append(ForestTypesDataset.create_forest_mask(sample_num))

    input_tensor = np.array(input_tensor)
    loaded_model = load_model(model, model_path)

    predict_mask = evaluate(loaded_model, input_tensor)

    if visualize:
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(input_tensor[0], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.imshow(predict_mask.clip(0.3, 0.75), cmap="gray")
        plt.subplot(1, 3, 3)
        plt.imshow(np.squeeze(ground_truth_tensor, axis=0), cmap="gray")
        plt.show()

    return predict_mask


def inference_test(
    model: nn.Module,
    model_path: Path,
    sample_num: str,
    num_runs: int,
    dataset_path: Path,
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
        input_tensor.append(ForestTypesDataset.create_forest_mask(sample_num))

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
