from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class ResNet50_RGB_NIR_fMASK_Dual_Model(nn.Module):
    def __init__(self, num_classes: int, freeze_encoder: bool = True):
        super(ResNet50_RGB_NIR_fMASK_Dual_Model, self).__init__()
        self.num_classes = num_classes
        self.logs_path = Path("./train_progress_v5")

        # Первый RGB энкодер ResNet
        self.encoder1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Второй RGB энкодер ResNet
        self.encoder2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        if freeze_encoder:
            self.freeze_rgb_layers()  # Вызываем заморозку сразу после загрузки весов

        # Модифицируем ResNet на 5 каналов на входе (RGB + NIR + маска)
        self.encoder1.conv1 = self.modify_first_layer(self.encoder1.conv1, in_channels=5)
        self.encoder1 = nn.Sequential(*list(self.encoder1.children())[:-2])  # убираем слой Average Pooling и FC
        self.encoder2.conv1 = self.modify_first_layer(self.encoder2.conv1, in_channels=5)
        self.encoder2 = nn.Sequential(*list(self.encoder2.children())[:-2])  # аналогично убираем ненужные слои

        # Декодер для объединения и разворачивания признаков в маску классов
        # 2048 выходных каналов у ResNet50, 4096 (2 * 2048) каналов, если объединены 2 энкодера
        self.decoder = self.build_decoder(2048 * 2)

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

    def freeze_rgb_layers(self):
        # Заморозка всех слоев энкодера для RGB из ResNet50
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False

    def build_decoder(self, in_channels):
        # Декодер с расширяющими слоями, возвращающий маску
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

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # Пропуск первого изображения через первый энкодер
        enc1_layers = []
        for layer in self.encoder1:
            x1 = layer(x1)
            enc1_layers.append(x1)
        # Пропуск второго изображения через второй энкодер
        enc2_layers = []
        for layer in self.encoder2:
            x2 = layer(x2)
            enc2_layers.append(x2)
        # Объединение признаков из двух энкодеров на самом глубоком уровне
        merged = torch.cat((enc1_layers[-1], enc2_layers[-1]), dim=1)
        # Пропуск через декодер для получения масок классов
        out = self.decoder(merged)
        # print("out:", out.shape)
        # exit()
        # Используем bilinear интерполяцию для приведения к исходному размеру
        # out = F.interpolate(out, size=(x1.shape[2], x1.shape[3]), mode="bilinear", align_corners=False)
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

    def train_model(
        self, train_dataset, val_dataset, epochs=1, batch_size=1, learning_rate=0.001, device="cuda", freeze_rgb=True
    ):
        self.to(device)
        criterion = nn.BCEWithLogitsLoss()  # Функция потерь для бинарной сегментации
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Размораживаем веса RGB-слоев
        if not freeze_rgb:
            for param in self.parameters():
                param.requires_grad = True

        for epoch in range(epochs):
            running_loss = 0.0
            total_samples = 0
            total_iou = 0.0
            total_accuracy = 0.0

            batch_inputs1, batch_inputs2, batch_masks = [], [], []

            for i, (sample1, sample2, mask) in enumerate(train_dataset.get_next_generated_sample(verbose=False)):
                # Подготовка данных для батча
                batch_inputs1.append(torch.tensor(sample1, dtype=torch.float32))
                batch_inputs2.append(torch.tensor(sample2, dtype=torch.float32))
                batch_masks.append(torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0))

                # Если собрали полный batch_size или это последний элемент
                if len(batch_inputs1) == batch_size or (i + 1) == len(train_dataset):
                    # Объединяем по batch размеру и перемещаем на устройство
                    input1_batch = torch.stack(batch_inputs1).to(device)
                    input2_batch = torch.stack(batch_inputs2).to(device)
                    mask_batch = torch.stack(batch_masks).to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    outputs: torch.Tensor = self(input1_batch, input2_batch)
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
                    batch_inputs1, batch_inputs2, batch_masks = [], [], []

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
            #     self.save_model(f"forest_resnet_snapshot_{epoch + 1}_{int(avg_loss * 1000)}.pth")

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

        batch_inputs1, batch_inputs2, batch_masks = [], [], []

        with torch.no_grad():
            for i, (sample1, sample2, mask) in enumerate(val_dataset.get_next_generated_sample(verbose=False)):
                # Prepare the data for batching
                batch_inputs1.append(torch.tensor(sample1, dtype=torch.float32))
                batch_inputs2.append(torch.tensor(sample2, dtype=torch.float32))
                batch_masks.append(torch.tensor(mask.copy(), dtype=torch.float32).unsqueeze(0))

                if len(batch_inputs1) == batch_size or (i + 1) == len(val_dataset):
                    # Convert batches to tensors and move to device
                    input1_batch = torch.stack(batch_inputs1).to(device)
                    input2_batch = torch.stack(batch_inputs2).to(device)
                    mask_batch = torch.stack(batch_masks).to(device)

                    # Forward pass
                    outputs: torch.Tensor = self(input1_batch, input2_batch)
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
                    batch_inputs1, batch_inputs2, batch_masks = [], [], []

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

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """На входе массивы с 5 каналами: (5, H, W) RGB + NIR + маска леса"""
        self.eval()  # Переключаем модель в режим оценки
        with torch.no_grad():
            outputs: torch.Tensor = self(self.prepare_input(x1), self.prepare_input(x2))
            return outputs.detach().cpu().squeeze().numpy()
