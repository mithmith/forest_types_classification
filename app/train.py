import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.loss import calculate_iou


def train_model(
    model: nn.Module, train_dataset, val_dataset, epochs=1, batch_size=1, learning_rate=0.001, device="cuda"
):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # Функция потерь для бинарной сегментации
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        running_loss = 0.0
        total_samples = 0
        total_iou = 0.0
        total_accuracy = 0.0

        batch_inputs, batch_masks = [], []

        for i, (sample, mask) in enumerate(train_dataset.get_next_generated_sample(verbose=False)):
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
                model_out_mask = outputs.detach().cpu().squeeze().numpy()
                plt.imshow(np.clip(model_out_mask, 0.3, 0.8), cmap="gray")
                plt.title("Model Output")

                plt.subplot(1, 2, 2)
                plt.imshow(mask_batch.cpu().squeeze().numpy(), cmap="gray")
                plt.title("Ground Truth Mask")

                if not self.logs_path.exists():
                    self.logs_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(self.logs_path / f"train_{epoch + 1}_{i + 1}_{int(avg_loss * 10000)}.png")
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

        validate(val_dataset, criterion, batch_size, device)

        # if (epoch + 1) % 5 == 0:
        # self.save_model(f"forest_resnet_snapshot_{epoch + 1}_{int(avg_loss * 1000)}.pth")

    print("Training complete")


def validate(model: nn.Module, val_dataset, criterion, batch_size: int, device: str):
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
        for i, (sample, mask) in enumerate(val_dataset.get_next_generated_sample(verbose=False)):
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
