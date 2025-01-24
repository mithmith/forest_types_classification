import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from app.loss import calculate_iou


def train_model(
    model: nn.Module, train_dataset, val_dataset, epochs=1, batch_size=1, learning_rate=0.001, device="cuda"
, exclude_nir=True, exclude_fMASK=True):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Функция потерь для бинарной сегментации
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        total_samples = 0
        total_iou = 0.0
        total_accuracy = 0.0

        batch_inputs, batch_masks = [], []

        for i, (sample, mask) in enumerate(train_dataset.get_next_generated_sample(verbose=False, exclude_nir=exclude_nir, exclude_fMASK=exclude_fMASK)):
            # Подготовка данных для батча
            batch_inputs.append(torch.tensor(sample, dtype=torch.float32))
            batch_masks.append(torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0))

            # Если собрали полный batch_size или это последний элемент
            if len(batch_inputs) == batch_size or (i + 1) == len(train_dataset):
                # Объединяем по batch размеру и перемещаем на устройство
                input_batch = torch.stack(batch_inputs).to(device)
                mask_batch = torch.stack(batch_masks).to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs: torch.Tensor = model(input_batch)
                loss = criterion(outputs, mask_batch)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_samples += 1

                # Calculate IoU
                iou = calculate_iou(torch.sigmoid(outputs), mask_batch)
                total_iou += iou

                # Calculate Overall Accuracy
                pred_mask = (torch.sigmoid(outputs) > 0.5).float()
                accuracy = (pred_mask == mask_batch).float().mean().item()
                total_accuracy += accuracy

                # Очищаем батчи для следующего набора
                batch_inputs, batch_masks = [], []

            if (i + 1) % 100 == 0:
                avg_loss = running_loss / total_samples
                avg_iou = total_iou / total_samples
                avg_accuracy = total_accuracy / total_samples
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataset)}],"
                    f" Average Loss: {avg_loss:.6f}, Average IoU: {avg_iou:.6f}, Avg Accuracy: {avg_accuracy:.6f}"
                )

                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                if batch_size > 1:
                    model_out_mask = outputs.detach().cpu().squeeze().numpy()[0]
                else:
                    model_out_mask = outputs.detach().cpu().squeeze().numpy()
                plt.imshow(np.clip(model_out_mask, 0.3, 0.8), cmap="gray")
                plt.title("Model Output")

                plt.subplot(1, 2, 2)
                if batch_size > 1:
                    plt.imshow(mask_batch.cpu().squeeze().numpy()[0], cmap="gray")
                else:
                    plt.imshow(mask_batch.cpu().squeeze().numpy(), cmap="gray")
                plt.title("Ground Truth Mask")

                if not model.logs_path.exists():
                    model.logs_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(model.logs_path / f"train_{epoch + 1}_{i + 1}_{int(avg_loss * 10000)}.png")
                # plt.show()
                plt.close("all")
            elif (i + 1) % 25 == 0:
                avg_loss = running_loss / total_samples
                avg_iou = total_iou / total_samples
                avg_accuracy = total_accuracy / total_samples
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataset)}],"
                    f" Average Loss: {avg_loss:.6f}, Average IoU: {avg_iou:.6f}, Avg Accuracy: {avg_accuracy:.6f}"
                )

        avg_loss = running_loss / total_samples
        avg_iou = total_iou / total_samples
        avg_accuracy = total_accuracy / total_samples

        print(
            f"Epoch [{epoch + 1}/{epochs}],"
            f" Average Loss: {avg_loss:.6f}, Average IoU: {avg_iou:.6f}, Avg Accuracy: {avg_accuracy:.6f}"
        )

        validate(model, val_dataset, criterion, batch_size, device, exclude_nir=exclude_nir, exclude_fMASK=exclude_fMASK)

        # if (epoch + 1) % 5 == 0:
        # self.save_model(f"forest_resnet_snapshot_{epoch + 1}_{int(avg_loss * 1000)}.pth")

    print("Training complete")


def validate(model: nn.Module, val_dataset, criterion, batch_size: int, device: str, exclude_nir=True, exclude_fMASK=True):
    """
    Validate the model using the validation dataset.
    Args:
        val_dataset (ForestTypesDataset): Dataset for validation.
        criterion: Loss function.
        batch_size (int): Batch size for validation.
        device (str): Device to use for validation ('cuda' or 'cpu').
    """
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    total_samples = 0
    total_iou = 0.0
    total_accuracy = 0.0

    batch_inputs, batch_masks = [], []

    with torch.no_grad():
        for i, (sample, mask) in enumerate(val_dataset.get_next_generated_sample(verbose=False, exclude_nir=exclude_nir, exclude_fMASK=exclude_fMASK)):
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

                # Clear batches
                batch_inputs, batch_masks = [], []

    avg_loss = running_loss / total_samples
    avg_iou = total_iou / total_samples
    avg_accuracy = total_accuracy / total_samples
    print(f"Validation Average Loss: {avg_loss:.6f}, Average IoU: {avg_iou:.6f}, Avg Accuracy: {avg_accuracy:.6f}")


def save_model(model, model_save_path: Path):
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


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
    model.to('cpu')
    with torch.no_grad():
        outputs: torch.Tensor = model(prepare_input(x1).to('cpu'))
        return outputs.detach().cpu().squeeze().numpy()
