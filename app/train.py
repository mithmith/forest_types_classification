from datetime import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt

plt.switch_backend("agg")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from app.loss import calculate_iou, iou_loss
from app.utils.veg_index import min_max_normalize_with_clipping


def train_model(
    model: nn.Module,
    train_dataset,
    val_dataset,
    epochs=1,
    batch_size=1,
    learning_rate=0.001,
    device="cuda",
    exclude_nir=True,
    exclude_fMASK=True,
    clearml_logger=None,
    model_name=None,
):
    model.to(device)
    # criterion = nn.BCEWithLogitsLoss()  # Функция потерь для бинарной сегментации
    criterion = iou_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.82)
    # logger.debug(f"Dataset length: {len(train_dataset)}")

    if not hasattr(model, "logs_path"):
        model.logs_path = Path("./logs")
    if not model.logs_path.exists():
        model.logs_path.mkdir(parents=True, exist_ok=True)

    logger.info("Model Architecture:")
    logger.info(model)

    num_train_samples = len(train_dataset)
    num_val_samples = len(val_dataset)
    loss_function_name = getattr(criterion, '__name__', str(criterion))
    optimizer_name = getattr(optimizer, '__name__', str(optimizer))
    scheduler_name = getattr(lr_scheduler, '__name__', str(lr_scheduler))
    logger.info(f"Number of training samples: {num_train_samples}")
    logger.info(f"Number of validation samples: {num_val_samples}")
    logger.info(f"Loss Function: {loss_function_name}")
    logger.info(f"Initial learning rate: {learning_rate}")
    logger.info(f"Optimizer Name: {optimizer_name}")
    logger.info(f"Scheduler Name: {scheduler_name}")

    sample_gen = train_dataset.get_next_generated_sample(
        verbose=False, exclude_nir=exclude_nir, exclude_fMASK=exclude_fMASK
    )

    sample, mask = next(sample_gen)
    sample_array = np.array(sample)
    mask_array = np.array(mask)
    logger.info(f"Sample input shape: {sample_array.shape}")
    logger.info(f"Sample mask shape: {mask_array.shape}")
    logger.info(f"Batch size: {batch_size}")

    if sample_array.shape[0] == 3:
        input_data_type = "RGB"
    elif sample_array.shape[0] == 4:
        input_data_type = "RGB+NIR"
    elif sample_array.shape[0] == 5:
        input_data_type = "RGB+NIR+fMASK"
    else:
        input_data_type = f"Unexpected input data type with {sample_array.shape[0]} channels"

    logger.info(f"Input data type: {input_data_type}")
    # Pass a sample through the model to obtain output shape.
    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(sample_tensor)
    logger.info(f"Model output shape: {output_tensor.shape}")

    total_foreground = 0
    total_pixels = 0
    sample_count = 0
    sample_gen_for_class = train_dataset.get_next_generated_sample(
        verbose=False, exclude_nir=exclude_nir, exclude_fMASK=exclude_fMASK
    )
    for s, m in sample_gen_for_class:
        m_arr = np.array(m)
        total_foreground += np.sum(m_arr > 0.5)
        total_pixels += m_arr.size
        sample_count += 1
        if sample_count >= 100:
            break
    if total_pixels > 0:
        foreground_percentage = (total_foreground / total_pixels) * 100
        logger.info(f"Approximate foreground percentage (from 100 samples): {foreground_percentage:.2f}%")

    logger.add(model.logs_path / f"{model_name}_training_logs.log", rotation="1 MB", level="DEBUG", format="{time} {level} {message}")

    iteration = 0
    global_iterations = []

    train_losses = []
    train_ious = []
    train_accuracies = []
    train_precisions = []

    val_losses = []
    val_ious = []
    val_accuracies = []
    val_precisions = []

    overall_start_time = time.time()

    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.requires_grad_(False)

    for epoch in range(epochs):
        running_loss = 0.0
        total_samples = 0
        total_iou = 0.0
        total_accuracy = 0.0
        total_precision = 0.0

        batch_inputs, batch_masks = [], []

        for i, (sample, mask) in enumerate(
            train_dataset.get_next_generated_sample(verbose=False, exclude_nir=exclude_nir, exclude_fMASK=exclude_fMASK)
        ):
            # Подготовка данных для батча
            batch_inputs.append(torch.tensor(sample, dtype=torch.float32))
            batch_masks.append(torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0))

            # Если собрали полный batch_size или это последний элемент
            if len(batch_inputs) == batch_size or (i + 1) == len(train_dataset):
                # Объединяем по batch размеру и перемещаем на устройство
                input_batch = torch.stack(batch_inputs).to(device)
                mask_batch = torch.stack(batch_masks).to(device)

                if torch.isnan(input_batch).any() or torch.isinf(input_batch).any():
                    logger.error("❌ ERROR: input_batch содержит NaN или inf!")
                    return False

                optimizer.zero_grad()

                # Forward pass
                outputs: torch.Tensor = model(input_batch)
                loss = criterion(outputs, mask_batch)

                # Backward pass and optimization
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         max_grad = param.grad.abs().max().item()
                #         if max_grad > 1e5:  # Градиенты слишком большие
                #             print(f"❌ WARNING: Взрыв градиентов в {name}: {max_grad}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        logger.warning(f"❌ WARNING: NaN в градиентах {name}")

                running_loss += loss.item()
                total_samples += 1

                # Calculate IoU
                iou = calculate_iou(torch.sigmoid(outputs), mask_batch)
                total_iou += iou
                # logger.debug(f"iou: {iou}, avr iou: {total_iou / total_samples}, total_samples: {total_samples}")

                # Calculate Overall Accuracy
                pred_mask = (torch.sigmoid(outputs) > 0.5).float()
                accuracy = (pred_mask == mask_batch).float().mean().item()
                total_accuracy += accuracy

                # Calculate Overall Precision
                pred_mask = (torch.sigmoid(outputs) > 0.5).float()
                tp = ((pred_mask == 1) & (mask_batch == 1)).float().sum().item()
                fp = ((pred_mask == 1) & (mask_batch == 0)).float().sum().item()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                total_precision += precision

                # Очищаем батчи для следующего набора
                batch_inputs, batch_masks = [], []

            if (i + 1) % 100 == 0:
                iteration += 1
                avg_loss = running_loss / total_samples
                avg_iou = total_iou / total_samples
                avg_accuracy = total_accuracy / total_samples
                avg_precision = total_precision / total_samples

                global_iterations.append(iteration)
                train_losses.append(avg_loss)
                train_ious.append(avg_iou)
                train_accuracies.append(avg_accuracy)
                train_precisions.append(avg_precision)

                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}], Iteration [{iteration}], "
                    f"Avg Loss: {avg_loss:.6f}, Avg IoU: {avg_iou:.6f}, "
                    f"Avg Accuracy: {avg_accuracy:.6f}, Avg Precision: {avg_precision:.6f}"
                )

                if clearml_logger is not None:
                    clearml_logger.current_logger().report_scalar(
                        title="Loss", series="Train Average Loss", value=avg_loss, iteration=iteration
                    )
                    clearml_logger.current_logger().report_scalar(
                        title="Metrics", series="Train Average IoU", value=avg_iou, iteration=iteration
                    )
                    clearml_logger.current_logger().report_scalar(
                        title="Metrics", series="Train Average Accuracy", value=avg_accuracy, iteration=iteration
                    )
                    clearml_logger.current_logger().report_scalar(
                        title="Metrics", series="Train Average Precision", value=avg_precision, iteration=iteration
                    )

                # Подготовка данных
                if batch_size > 1:
                    model_out_mask = outputs.detach().cpu().squeeze().numpy()[0]  # Предсказанная маска
                    ground_truth_mask = mask_batch.cpu().squeeze().numpy()[0]  # Истинная маска
                    rgb_image = input_batch.cpu().squeeze().numpy()[0, :3]  # Первые три слоя (RGB)
                else:
                    model_out_mask = outputs.detach().cpu().squeeze().numpy()
                    ground_truth_mask = mask_batch.cpu().squeeze().numpy()
                    rgb_image = input_batch.cpu().squeeze().numpy()[:3]

                # Нормализация RGB-данных для визуализации
                # logger.debug(f"Размерность rgb_image перед обработкой: {rgb_image.shape}")
                normalized_rgb = min_max_normalize_with_clipping(np.transpose(rgb_image, (1, 2, 0)))
                predicted_mask = (np.clip(model_out_mask, 0, 1) > 0.5).astype(np.uint8)
                rgb_uint8 = np.clip(normalized_rgb * 255, 0, 255).astype(np.uint8)

                # Построение визуализаций
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 3, 1)
                plt.imshow(rgb_uint8)
                plt.imshow(predicted_mask, cmap="hot", alpha=0.5)
                plt.title("RGB Image + Model Mask")
                plt.subplot(1, 3, 2)
                plt.imshow(rgb_uint8)
                plt.imshow(ground_truth_mask, cmap="hot", alpha=0.5)
                plt.title("RGB Image + Ground Truth Mask")
                plt.subplot(1, 3, 3)
                plt.imshow(ground_truth_mask, cmap="gray")
                plt.imshow(predicted_mask, cmap="hot", alpha=0.5)
                plt.title("Ground Truth Mask + Model Mask")
                plt.tight_layout()

                # Сохранение изображения
                if not model.logs_path.exists():
                    model.logs_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(model.logs_path / f"train_{epoch + 1}_{iteration}_{int(avg_loss * 10000)}.png")
                # plt.show()
                plt.close("all")
            elif (i + 1) % 25 == 0:
                avg_loss = running_loss / total_samples
                avg_iou = total_iou / total_samples
                avg_accuracy = total_accuracy / total_samples
                avg_precision = total_precision / total_samples

                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataset)}], "
                    f"Avg Loss: {avg_loss:.6f}, Avg IoU: {avg_iou:.6f}, "
                    f"Avg Accuracy: {avg_accuracy:.6f}, Avg Precision: {avg_precision:.6f}"
                )

        avg_loss = running_loss / total_samples
        avg_iou = total_iou / total_samples
        avg_accuracy = total_accuracy / total_samples
        avg_precision = total_precision / total_samples

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] Summary: Avg Loss: {avg_loss:.6f}, "
            f"Avg IoU: {avg_iou:.6f}, Avg Accuracy: {avg_accuracy:.6f}, Avg Precision: {avg_precision:.6f}"
        )

        val_metrics = validate(
            model,
            val_dataset,
            criterion,
            batch_size,
            device,
            exclude_nir=exclude_nir,
            exclude_fMASK=exclude_fMASK,
            clearml_logger=clearml_logger,
            iteration=iteration,
        )

        model.train()

        val_losses.append(val_metrics.get("loss", 0.0))
        val_ious.append(val_metrics.get("iou", 0.0))
        val_accuracies.append(val_metrics.get("accuracy", 0.0))
        val_precisions.append(val_metrics.get("precision", 0.0))

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Updated Learning Rate: {current_lr:.6f}")

        if (epoch + 1) % 10 == 0:
            save_model(model, f"{model_name}_snapshot_epoch_{epoch + 1}.pth")

    overall_end_time = time.time()
    total_training_time = overall_end_time - overall_start_time

    plot_metrics(
        global_iterations,
        train_losses, val_losses,
        train_ious, val_ious,
        train_accuracies, val_accuracies,
        train_precisions, val_precisions,
        model.logs_path, model_name, loss_function_name,
        epochs
    )

    final_summary = {
        "Model Architecture": str(model),
        "Input Sample Shape": str(sample_array.shape),
        "Input data type": str(input_data_type),
        "Model Output Shape": str(output_tensor.shape),
        "Batch Size": batch_size,
        "Loss Function": loss_function_name,
        "Initial Learning Rate": learning_rate,
        "Optimizer Name": {optimizer_name},
        "Scheduler Name": {scheduler_name},
        "Number of Training Samples": num_train_samples,
        "Number of Validation Samples": num_val_samples,
        "Final Training Avg Loss": round(train_losses[-1], 6),
        "Final Training Avg IoU": round(train_ious[-1], 6),
        "Final Training Avg Accuracy": round(train_accuracies[-1], 6),
        "Final Training Avg Precision": round(train_precisions[-1], 6),
        "Final Validation Loss": round(val_losses[-1], 6),
        "Final Validation IoU": round(val_ious[-1], 6),
        "Final Validation Accuracy": round(val_accuracies[-1], 6),
        "Final Validation Precision": round(val_precisions[-1], 6),
        "Total Training Time (hours)": f"{total_training_time / 3600:.2f}",
        "Average Time per Epoch (hours)": f"{(total_training_time / 3600) / epochs:.2f}",
    }
    save_model_summary(final_summary, model.logs_path, model_name)

    logger.info("Training complete")
    logger.info(f"Total Training Time (hours): {total_training_time / 3600:.2f}")


def validate(
    model: nn.Module,
    val_dataset,
    criterion,
    batch_size: int,
    device: str,
    exclude_nir=True,
    exclude_fMASK=True,
    clearml_logger=None,
    iteration=None,
):
    """
    Validate the model using the validation dataset.
    Args:
        val_dataset: Dataset for validation.
        criterion: Loss function.
        batch_size (int): Batch size for validation.
        device (str): Device to use for validation ('cuda' or 'cpu').
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    total_samples = 0
    total_iou = 0.0
    total_accuracy = 0.0
    total_precision = 0.0

    batch_inputs, batch_masks = [], []

    with torch.no_grad():
        for i, (sample, mask) in enumerate(
            val_dataset.get_next_generated_sample(verbose=False, exclude_nir=exclude_nir, exclude_fMASK=exclude_fMASK)
        ):
            # Prepare the data for batching
            batch_inputs.append(torch.tensor(sample, dtype=torch.float32))
            batch_masks.append(torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0))

            if len(batch_inputs) == batch_size or (i + 1) == len(val_dataset):
                # Convert batches to tensors and move to device
                input_batch = torch.stack(batch_inputs).to(device)
                mask_batch = torch.stack(batch_masks).to(device)

                # Forward pass
                outputs = model(input_batch)
                loss = criterion(outputs, mask_batch)

                running_loss += loss.item()
                total_samples += 1

                # Calculate IoU
                iou = calculate_iou(torch.sigmoid(outputs), mask_batch)
                total_iou += iou

                # Calculate Overall Accuracy
                pred_mask = (torch.sigmoid(outputs) > 0.5).float()
                accuracy = (pred_mask == mask_batch).float().mean().item()
                total_accuracy += accuracy

                # Calculate Overall Precision
                pred_mask = (torch.sigmoid(outputs) > 0.5).float()
                tp = ((pred_mask == 1) & (mask_batch == 1)).float().sum().item()
                fp = ((pred_mask == 1) & (mask_batch == 0)).float().sum().item()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                total_precision += precision

                # Clear batches
                batch_inputs, batch_masks = [], []

    avg_loss = running_loss / total_samples
    avg_iou = total_iou / total_samples
    avg_accuracy = total_accuracy / total_samples
    avg_precision = total_precision / total_samples

    if clearml_logger is not None:
        clearml_logger.current_logger().report_scalar(
            title="Loss", series="Validation Average Loss", value=avg_loss, iteration=iteration
        )
        clearml_logger.current_logger().report_scalar(
            title="Metrics", series="Validation Average IoU", value=avg_iou, iteration=iteration
        )
        clearml_logger.current_logger().report_scalar(
            title="Metrics", series="Validation Average Accuracy", value=avg_accuracy, iteration=iteration
        )
        clearml_logger.current_logger().report_scalar(
            title="Metrics", series="Validation Average Precision", value=avg_precision, iteration=iteration
        )

    logger.info(
        f"Validation Metrics: Avg Loss: {avg_loss:.6f}, Avg IoU: {avg_iou:.6f}, "
        f"Avg Accuracy: {avg_accuracy:.6f}, Avg Precision: {avg_precision:.6f}"
    )

    return {
        "loss": avg_loss,
        "iou": avg_iou,
        "accuracy": avg_accuracy,
        "precision": avg_precision,
    }


def save_model(model, model_save_path: Path):
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")


def load_model(model, model_load_path: Path):
    model.load_state_dict(torch.load(model_load_path))
    print(f"Model loaded from {model_load_path}")
    return model


@staticmethod
def prepare_input(combined_array: np.ndarray) -> torch.Tensor:
    """
    Функция для подготовки тензора с 5 каналами (5, H, W) (RGB + NIR + маска).

    Аргументы:
    - red, green, blue, nir, mask: numpy массивы формы (H, W) для каждого канала

    Возвращает:
    - torch.Tensor формы (1, 5, H, W) для подачи в модель
    """
    # Преобразуем в тензор и добавляем batch размерности (1, 5, H, W)
    return torch.tensor(combined_array, dtype=torch.float32).unsqueeze(0)


def evaluate(model, x1: np.ndarray) -> np.ndarray:
    model.eval()  # Переключаем модель в режим оценки
    model.to("cpu")
    with torch.no_grad():
        outputs: torch.Tensor = model(prepare_input(x1).to("cpu"))
        return outputs.detach().cpu().squeeze().numpy()


def save_model_summary(summary: dict, logs_path: Path, model_name: str):
    """Save model summary information to a text file."""
    summary_file = logs_path / f"{model_name}_model_summary.txt"
    with open(summary_file, "w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    logger.info(f"Training summary saved to {summary_file}")


def plot_metrics(iterations, train_losses, val_losses,
                 train_ious, val_ious,
                 train_accuracies, val_accuracies,
                 train_precisions, val_precisions,
                 logs_path: Path, model_name: str, loss_name: str,
                 epochs):
    """Plot epoch-level training and validation curves in one figure."""
    plt.figure(figsize=(12, 10))
    iterations_range = range(1, iterations[-1] + 1)
    val_iterations_range = range(int(iterations[-1]//epochs), iterations[-1] + 1, int(iterations[-1]//epochs))

    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(iterations_range, train_losses, label=f"Train Loss ({loss_name})", marker="o")
    plt.plot(val_iterations_range, val_losses, label=f"Validation Loss ({loss_name})", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid()

    # IoU plot
    plt.subplot(2, 2, 2)
    plt.plot(iterations_range, train_ious, label="Train IoU", marker="o")
    plt.plot(val_iterations_range, val_ious, label="Validation IoU", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("IoU")
    plt.title("IoU")
    plt.legend()
    plt.grid()

    # Accuracy plot
    plt.subplot(2, 2, 3)
    plt.plot(iterations_range, train_accuracies, label="Train Accuracy", marker="o")
    plt.plot(val_iterations_range, val_accuracies, label="Validation Accuracy", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid()

    # Precision plot
    plt.subplot(2, 2, 4)
    plt.plot(iterations_range, train_precisions, label="Train Precision", marker="o")
    plt.plot(val_iterations_range, val_precisions, label="Validation Precision", marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Precision")
    plt.title("Precision")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    metrics_plot_file = logs_path / f"{model_name}_training_metrics.png"
    plt.savefig(metrics_plot_file)
    plt.close()
    logger.info(f"Saved training metrics plot to {metrics_plot_file}")

