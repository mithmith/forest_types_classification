import os
import random
from pathlib import Path
from clearml import Task

from app.dataset_single import ForestTypesDataset
from app.modelResNet50_RGB import ResNet50_RGB_Model

os.environ["GDAL_DATA"] = os.environ["CONDA_PREFIX"] + r"\Library\share\gdal"
os.environ["PROJ_LIB"] = os.environ["CONDA_PREFIX"] + r"\Library\share"
os.environ["GDAL_DRIVER_PATH"] = os.environ["CONDA_PREFIX"] + r"\Library\lib\gdalplugins"

# Prefixes
path_prefix = Path("G:/Orni_forest/")
damage_prefix = "_drying"

# Dataset paths
dataset_geojson_masks_dir = path_prefix.joinpath(f"forest_changes_dataset/masks{damage_prefix}")
sentinel_root_dir = path_prefix.joinpath(f"forest_changes_dataset/images{damage_prefix}")
train_path = path_prefix.joinpath(f"forest_changes_dataset/generated_dataset{damage_prefix}_RGBNIRSWIR/train")
val_path = path_prefix.joinpath(f"forest_changes_dataset/generated_dataset{damage_prefix}_RGBNIRSWIR/validation")

train_dataset = ForestTypesDataset(
    dataset_geojson_masks_dir,
    sentinel_root_dir,
    dataset_path=train_path,
)
val_dataset = ForestTypesDataset(
    dataset_geojson_masks_dir,
    sentinel_root_dir,
    dataset_path=val_path,
)

# Version and model
i = 1
model = ResNet50_RGB_Model(num_classes=1)
model_name = 'ResNet50_RGB_Model' + damage_prefix

# Training progresses and models paths
training_process_path = path_prefix.joinpath(f"sentinel_forest_types_classification_training_process/{model_name}/train_progress_v{i}/")
model_save_path = path_prefix.joinpath(f"sentinel_forest_types_classification_models/{model_name}/")
model_load_path = model_save_path.joinpath(f"{model_name}_v{i-1}.pth")

for path in [training_process_path, model_save_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Training
task = Task.init(project_name=f'ML-{model_name}', task_name=f'Forest_changes_v{i}', reuse_last_task_id=False)

# model.load_model(model_load_path)
model.logs_path = training_process_path
model.train_model(train_dataset=train_dataset, val_dataset=val_dataset, epochs=10, batch_size=1, learning_rate=0.001)
model.save_model(model_save_path.joinpath(f"{model_name}_v{i}.pth"))

# Inference test
red_tif_files = list(val_path.glob("*_red.tif"))
random.shuffle(red_tif_files)
for filename in red_tif_files[:5]:
    n = filename.stem.split("_")[0]
    val_dataset.inference_test(model, model_save_path.joinpath(f"{model_name}_v{i}.pth"), n, 100, exclude_nir=True, exclude_fMASK=True)

task.close()
