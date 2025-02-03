import os
import random
from pathlib import Path

from clearml import Task

from app.dataset_single import ForestTypesDataset
from app.modelMobileNetV3_UNet import MobileNetV3_UNet
from app.modelMobileNetV3_UNet_NIR import MobileNetV3_UNet_NIR
from app.modelMobileNetV3_UNet_NIR_fMASK import MobileNetV3_UNet_NIR_fMASK
from app.modelResNet50_RGB import ResNet50_RGB_Model, ResNet50_UNet
from app.modelResNet50_RGB_NIR import ResNet50_RGB_NIR_Model, ResNet50_UNet_NIR
from app.modelResNet50_RGB_NIR_fMASK import ResNet50_RGB_NIR_fMASK_Model, ResNet50_UNet_NIR_fMASK
from app.modelSKResNeXt50_UNet import SKResNeXt50_UNet
from app.modelSKResNeXt50_UNet_NIR import SKResNeXt50_UNet_NIR
from app.modelSKResNeXt50_UNet_NIR_fMASK import SKResNeXt50_UNet_NIR_fMASK
from app.MobileNetV3_PSPNet_RGB import MobileNetV3_PSPNet
from app.ResNet50_PSPNet_RGB import ResNet50_PSPNet
from app.SKResNeXt50_PSPNet_RGB import SKResNeXt50_PSPNet
from app.train import save_model, train_model, load_model
import evaluate

# os.environ["GDAL_DATA"] = os.environ["CONDA_PREFIX"] + r"\Library\share\gdal"
# os.environ["PROJ_LIB"] = os.environ["CONDA_PREFIX"] + r"\Library\share"
# os.environ["GDAL_DRIVER_PATH"] = os.environ["CONDA_PREFIX"] + r"\Library\lib\gdalplugins"

# Prefixes
path_prefix = Path("G:/Orni_forest/")
damage_prefixes = ["_burns", "_deforestation", "drying"]

for damage_prefix in damage_prefixes:
    # Dataset paths
    dataset_geojson_masks_dir = path_prefix.joinpath(f"forest_changes_dataset/masks{damage_prefix}")
    sentinel_root_dir = path_prefix.joinpath(f"forest_changes_dataset/images{damage_prefix}")
    crop_bboxes_dir = path_prefix.joinpath(f"forest_changes_dataset/crop_bboxes")
    train_path = path_prefix.joinpath(f"forest_changes_dataset/generated_dataset{damage_prefix}_RGBNIRSWIR/train")
    val_path = path_prefix.joinpath(f"forest_changes_dataset/generated_dataset{damage_prefix}_RGBNIRSWIR/validation")

    forest_model_path = Path("../forest_model_v8.dat")

    train_dataset = ForestTypesDataset(
        dataset_geojson_masks_dir,
        sentinel_root_dir,
        crop_bboxes_dir=crop_bboxes_dir,
        dataset_path=train_path,
        forest_model_path=forest_model_path,
    )
    val_dataset = ForestTypesDataset(
        dataset_geojson_masks_dir,
        sentinel_root_dir,
        crop_bboxes_dir=crop_bboxes_dir,
        dataset_path=val_path,
        forest_model_path=forest_model_path,
    )

    # Generate dataset
    # train_dataset.generate_dataset(4500)
    # exit()

    # Version and model
    i = 1
    freezed = True

    model_MobileNetV3_UNet = MobileNetV3_UNet(num_classes=1, freeze_encoder=freezed)
    model_MobileNetV3_UNet_NIR = MobileNetV3_UNet_NIR(num_classes=1, freeze_encoder=freezed)
    model_MobileNetV3_UNet_NIR_fMASK = MobileNetV3_UNet_NIR_fMASK(num_classes=1, freeze_encoder=freezed)

    model_ResNet50_UNet = ResNet50_UNet(num_classes=1, freeze_encoder=freezed)
    model_ResNet50_UNet_NIR = ResNet50_UNet_NIR(num_classes=1, freeze_encoder=freezed)
    model_ResNet50_UNet_NIR_fMASK = ResNet50_UNet_NIR_fMASK(num_classes=1, freeze_encoder=freezed)

    model_ResNet50_RGB_Model = ResNet50_RGB_Model(num_classes=1)
    model_ResNet50_RGB_NIR_Model = ResNet50_RGB_NIR_Model(num_classes=1)
    model_ResNet50_RGB_NIR_fMASK_Model = ResNet50_RGB_NIR_fMASK_Model(num_classes=1)

    model_SKResNeXt50_UNet = SKResNeXt50_UNet(num_classes=1, freeze_encoder=freezed)
    model_SKResNeXt50_UNet_NIR = SKResNeXt50_UNet_NIR(num_classes=1, freeze_encoder=freezed)
    model_SKResNeXt50_UNet_NIR_fMASK = SKResNeXt50_UNet_NIR_fMASK(num_classes=1, freeze_encoder=freezed)

    model_MobileNetV3_PSPNet = MobileNetV3_PSPNet(num_classes=1, freeze_encoder=freezed)
    model_ResNet50_PSPNet = ResNet50_PSPNet(num_classes=1, freeze_encoder=freezed)
    model_SKResNeXt50_PSPNet = SKResNeXt50_PSPNet(num_classes=1, freeze_encoder=freezed)

    if freezed:
        freez_prefix = "_freezed"
    else:
        freez_prefix = "_unfreezed"

    models = {
        "MobileNetV3_PSPNet" + damage_prefix + freez_prefix: model_MobileNetV3_PSPNet,
        "ResNet50_PSPNet" + damage_prefix + freez_prefix: model_ResNet50_PSPNet,
        "SKResNeXt50_PSPNet" + damage_prefix + freez_prefix: model_SKResNeXt50_PSPNet,
        # "MobileNetV3_UNet" + damage_prefix + freez_prefix: model_MobileNetV3_UNet,
        # "MobileNetV3_UNet_NIR" + damage_prefix + freez_prefix: model_MobileNetV3_UNet_NIR,
        # "MobileNetV3_UNet_NIR_fMASK" + damage_prefix + freez_prefix: model_MobileNetV3_UNet_NIR_fMASK,
        # "ResNet50_UNet" + damage_prefix + freez_prefix: model_ResNet50_UNet,
        # "ResNet50_UNet_NIR" + damage_prefix + freez_prefix: model_ResNet50_UNet_NIR,
        # "ResNet50_UNet_NIR_fMASK" + damage_prefix + freez_prefix: model_ResNet50_UNet_NIR_fMASK,
        # "ResNet50_RGB_Model" + damage_prefix + freez_prefix: model_ResNet50_RGB_Model,
        # "ResNet50_RGB_NIR_Model" + damage_prefix + freez_prefix: model_ResNet50_RGB_NIR_Model,
        # "ResNet50_RGB_NIR_fMASK_Model" + damage_prefix + freez_prefix: model_ResNet50_RGB_NIR_fMASK_Model,
        # "SKResNeXt50_UNet" + damage_prefix + freez_prefix: model_SKResNeXt50_UNet,
        # "SKResNeXt50_UNet_NIR" + damage_prefix + freez_prefix: model_SKResNeXt50_UNet_NIR,
        # "SKResNeXt50_UNet_NIR_fMASK" + damage_prefix + freez_prefix: model_SKResNeXt50_UNet_NIR_fMASK,
    }

    for model_name, model in models.items():

        if "NIR" in model_name and "fMASK" in model_name:
            exclude_nir = False
            exclude_fMASK = False
        elif "NIR" in model_name:
            exclude_nir = False
            exclude_fMASK = True
        else:
            exclude_nir = True
            exclude_fMASK = True

        # Training progresses and models paths
        training_process_path = (
            path_prefix / f"forest_types_classification_training_process/{model_name}/train_progress_v{i}/"
        )
        model_save_path = path_prefix / f"forest_types_classification_models/{model_name}/"
        model_load_path = model_save_path / f"{model_name}_v{i-1}.pth"
        evaluation_dir = training_process_path / "evaluation_results/"

        for path in [training_process_path, model_save_path, evaluation_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        # Training
        task = Task.init(project_name=f"ML-{model_name}", task_name=f"Forest_changes_v{i}", reuse_last_task_id=False)
        clearml_logger = task.get_logger()

        # model = load_model(model, model_load_path)
        model.logs_path = training_process_path
        train_model(
            model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=10,
            batch_size=2,
            learning_rate=0.001,
            exclude_nir=exclude_nir,
            exclude_fMASK=exclude_fMASK,
            clearml_logger=clearml_logger,
        )
        save_model(model, model_save_path.joinpath(f"{model_name}_v{i}.pth"))

        # Inference test
        red_tif_files = list(val_path.glob("*_red.tif"))
        random.shuffle(red_tif_files)
        for filename in red_tif_files[:5]:
            n = filename.stem.split("_")[0]

            evaluate.inference_test(
                model,
                model_save_path.joinpath(f"{model_name}_v{i}.pth"),
                n,
                100,
                val_path,
                forest_model_path,
                exclude_nir=exclude_nir,
                exclude_fMASK=exclude_fMASK,
            )

            evaluate.predict_sample_from_dataset(
                model,
                model_save_path.joinpath(f"{model_name}_v{i}.pth"),
                n,
                val_path,
                forest_model_path,
                exclude_nir=exclude_nir,
                exclude_fMASK=exclude_fMASK,
                evaluation_dir=evaluation_dir,
                file_name=model_name,
            )

        task.close()
