from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from clearml import Task

from app.dataset_single import ForestTypesDataset


class ResNet50_RGB_NIR_Model(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet50_RGB_NIR_Model, self).__init__()
        self.num_classes = num_classes
        self.logs_path = Path("./train_progress_unet")

        # Single ResNet encoder
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.encoder.conv1 = self.modify_first_layer(self.encoder.conv1, in_channels=4)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])  # Remove Average Pooling and FC layers

        # Decoder
        self.decoder = self.build_decoder(2048)

    def modify_first_layer(self, conv, in_channels: int):
        # Модификация первого свёрточного слоя для обработки 5 каналов
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )

        # Инициализация весов для первых 3 каналов из предобученной модели
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = conv.weight  # копируем RGB
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode="fan_out", nonlinearity="relu")
        return new_conv

    def build_decoder(self, in_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_classes, kernel_size=1),  # Выходной слой для маски классов
        )

    def forward(self, x: torch.Tensor):
        # Encoder forward pass
        enc_features = []
        for layer in self.encoder:
            x = layer(x)
            enc_features.append(x)

        # Decoder forward pass
        out = self.decoder(enc_features[-1])

        # Bilinear interpolation to match input size
        # return nn.functional.interpolate(out, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        return out

    def calculate_iou(self, pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Calculate Intersection over Union (IoU) for binary masks.
        :param pred_mask: Predicted mask (logits or probabilities).
        :param true_mask: Ground truth mask.
        :param threshold: Threshold to binarize the predicted mask.
        :return: IoU score.
        """
        pred_mask = (pred_mask > threshold).float()  # Binarize predicted mask
        intersection = (pred_mask * true_mask).sum()  # Intersection area
        union = pred_mask.sum() + true_mask.sum() - intersection  # Union area
        iou = intersection / union if union > 0 else 0.0  # IoU formula
        return iou.item()

    def train_model(self, train_dataset, val_dataset, epochs=1, batch_size=1, learning_rate=0.001, device="cuda"):
        self.to(device)
        criterion = nn.BCEWithLogitsLoss()  # Функция потерь для бинарной сегментации
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            running_loss = 0.0
            total_samples = 0
            total_iou = 0.0
            total_accuracy = 0.0

            batch_inputs, batch_masks = [], []

            for i, (sample, mask) in enumerate(
                train_dataset.get_next_generated_sample(verbose=False, exclude_nir=False)
            ):
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
                    outputs: torch.Tensor = self(input_batch)
                    loss = criterion(outputs, mask_batch)

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    total_samples += 1

                    # Calculate IoU
                    iou = self.calculate_iou(torch.sigmoid(outputs), mask_batch)
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

            self.validate(val_dataset, criterion, batch_size, device)

            # if (epoch + 1) % 5 == 0:
            # self.save_model(f"forest_resnet_snapshot_{epoch + 1}_{int(avg_loss * 1000)}.pth")

        print("Training complete")

    def validate(self, val_dataset, criterion, batch_size: int, device: str):
        """
        Validate the model using the validation dataset.
        Args:
            val_dataset (ForestTypesDataset): Dataset for validation.
            criterion: Loss function.
            batch_size (int): Batch size for validation.
            device (str): Device to use for validation ('cuda' or 'cpu').
        """
        self.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        total_samples = 0
        total_iou = 0.0
        total_accuracy = 0.0

        batch_inputs, batch_masks = [], []

        with torch.no_grad():
            for i, (sample, mask) in enumerate(val_dataset.get_next_generated_sample(verbose=False, exclude_nir=False)):
                # Prepare the data for batching
                batch_inputs.append(torch.tensor(sample, dtype=torch.float32))
                batch_masks.append(torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0))

                if len(batch_inputs) == batch_size or (i + 1) == len(val_dataset):
                    # Convert batches to tensors and move to device
                    input_batch = torch.stack(batch_inputs).to(device)
                    mask_batch = torch.stack(batch_masks).to(device)

                    # Forward pass
                    outputs = self(input_batch)
                    loss = criterion(outputs, mask_batch)

                    running_loss += loss.item()
                    total_samples += 1

                    # Calculate IoU
                    iou = self.calculate_iou(torch.sigmoid(outputs), mask_batch)
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

    def save_model(self, model_save_path: Path):
        torch.save(self.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    def load_model(self, model_load_path: Path):
        self.load_state_dict(torch.load(model_load_path))
        print(f"Model loaded from {model_load_path}")

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

    def evaluate(self, x1: np.ndarray) -> np.ndarray:
        self.eval()  # Переключаем модель в режим оценки
        with torch.no_grad():
            outputs: torch.Tensor = self(self.prepare_input(x1))
            return outputs.detach().cpu().squeeze().numpy()


class ResNet50_UNet_NIR(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet50_UNet_NIR, self).__init__()
        self.num_classes = num_classes

        # Single ResNet encoder
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.freeze_rgb_layers()  # Вызываем заморозку сразу после загрузки весов

        self.encoder.conv1 = self.modify_first_layer(self.encoder.conv1, in_channels=4)

        # Save intermediate features for skip connections
        self.enc1 = nn.Sequential(*list(self.encoder.children())[:3])  # Conv1 + BN + ReLU
        self.enc2 = nn.Sequential(*list(self.encoder.children())[3:5])  # MaxPool + Layer1
        self.enc3 = self.encoder.layer2
        self.enc4 = self.encoder.layer3
        self.enc5 = self.encoder.layer4

        # Decoder with skip connections
        self.dec4 = self.build_decoder_block(2048, 1024)
        self.dec3 = self.build_decoder_block(1024 + 1024, 512)
        self.dec2 = self.build_decoder_block(512 + 512, 256)
        self.dec1 = self.build_decoder_block(256 + 256, 128)
        self.dec0 = self.build_decoder_block(128 + 64, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, self.num_classes, kernel_size=1)

    def modify_first_layer(self, conv, in_channels: int):
        """Modify the first convolution layer to accept more channels."""
        new_conv = nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = conv.weight  # Copy RGB weights
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode="fan_out", nonlinearity="relu")
        return new_conv

    def build_decoder_block(self, in_channels, out_channels):
        """Build a single block of the decoder."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def freeze_rgb_layers(self):
        # Заморозка всех слоев энкодера для RGB из ResNet50
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)  # 512x512 -> 256x256
        enc2_out = self.enc2(enc1_out)  # 256x256 -> 128x128
        enc3_out = self.enc3(enc2_out)  # 128x128 -> 64x64
        enc4_out: torch.Tensor = self.enc4(enc3_out)  # 64x64 -> 32x32
        enc5_out: torch.Tensor = self.enc5(enc4_out)  # 32x32 -> 16x16

        # Decoder with skip connections
        dec4_out = self.dec4(enc5_out)  # 16x16 -> 32x32
        dec3_out = self.dec3(torch.cat([dec4_out, enc4_out], dim=1))  # 32x32 -> 64x64
        dec2_out = self.dec2(torch.cat([dec3_out, enc3_out], dim=1))  # 64x64 -> 128x128
        dec1_out = self.dec1(torch.cat([dec2_out, enc2_out], dim=1))  # 128x128 -> 256x256
        output = self.dec0(torch.cat([dec1_out, enc1_out], dim=1))  # 256x256 -> 512x512
        return self.final_conv(output)

    def calculate_iou(self, pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Calculate Intersection over Union (IoU) for binary masks.
        :param pred_mask: Predicted mask (logits or probabilities).
        :param true_mask: Ground truth mask.
        :param threshold: Threshold to binarize the predicted mask.
        :return: IoU score.
        """
        pred_mask = (pred_mask > threshold).float()  # Binarize predicted mask
        intersection = (pred_mask * true_mask).sum()  # Intersection area
        union = pred_mask.sum() + true_mask.sum() - intersection  # Union area
        iou = intersection / union if union > 0 else 0.0  # IoU formula
        return iou.item()

    def train_model(self, train_dataset, val_dataset, epochs=1, batch_size=1, learning_rate=0.001, device="cuda"):
        self.to(device)
        criterion = nn.BCEWithLogitsLoss()  # Функция потерь для бинарной сегментации
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            running_loss = 0.0
            total_samples = 0
            total_iou = 0.0
            total_accuracy = 0.0

            batch_inputs, batch_masks = [], []

            for i, (sample, mask) in enumerate(
                train_dataset.get_next_generated_sample(verbose=False, exclude_nir=False)
            ):
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
                    outputs: torch.Tensor = self(input_batch)
                    loss = criterion(outputs, mask_batch)

                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    total_samples += 1

                    # Calculate IoU
                    iou = self.calculate_iou(torch.sigmoid(outputs), mask_batch)
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

            self.validate(val_dataset, criterion, batch_size, device)

            # if (epoch + 1) % 5 == 0:
            # self.save_model(f"forest_resnet_snapshot_{epoch + 1}_{int(avg_loss * 1000)}.pth")

        print("Training complete")

    def validate(self, val_dataset, criterion, batch_size: int, device: str):
        """
        Validate the model using the validation dataset.
        Args:
            val_dataset (ForestTypesDataset): Dataset for validation.
            criterion: Loss function.
            batch_size (int): Batch size for validation.
            device (str): Device to use for validation ('cuda' or 'cpu').
        """
        self.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        total_samples = 0
        total_iou = 0.0
        total_accuracy = 0.0

        batch_inputs, batch_masks = [], []

        with torch.no_grad():
            for i, (sample, mask) in enumerate(val_dataset.get_next_generated_sample(verbose=False, exclude_nir=False)):
                # Prepare the data for batching
                batch_inputs.append(torch.tensor(sample, dtype=torch.float32))
                batch_masks.append(torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0))

                if len(batch_inputs) == batch_size or (i + 1) == len(val_dataset):
                    # Convert batches to tensors and move to device
                    input_batch = torch.stack(batch_inputs).to(device)
                    mask_batch = torch.stack(batch_masks).to(device)

                    # Forward pass
                    outputs = self(input_batch)
                    loss = criterion(outputs, mask_batch)

                    running_loss += loss.item()
                    total_samples += 1

                    # Calculate IoU
                    iou = self.calculate_iou(torch.sigmoid(outputs), mask_batch)
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

    def save_model(self, model_save_path: Path):
        torch.save(self.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    def load_model(self, model_load_path: Path):
        self.load_state_dict(torch.load(model_load_path))
        print(f"Model loaded from {model_load_path}")

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

    def evaluate(self, x1: np.ndarray) -> np.ndarray:
        self.eval()  # Переключаем модель в режим оценки
        with torch.no_grad():
            outputs: torch.Tensor = self(self.prepare_input(x1))
            return outputs.detach().cpu().squeeze().numpy()
