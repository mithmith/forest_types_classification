from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from app.dataset_single import ForestTypesDataset


class ResNet50_RGB_NIR_fMASK_Model(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet50_RGB_NIR_fMASK_Model, self).__init__()
        self.num_classes = num_classes
        self.logs_path = Path("./train_progress_unet")

        # Single ResNet encoder
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.encoder.conv1 = self.modify_first_layer(self.encoder.conv1, in_channels=5)
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


class ResNet50_UNet_NIR_fMASK(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNet50_UNet_NIR_fMASK, self).__init__()
        self.num_classes = num_classes

        # Single ResNet encoder
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.freeze_rgb_layers()  # Вызываем заморозку сразу после загрузки весов

        self.encoder.conv1 = self.modify_first_layer(self.encoder.conv1, in_channels=5)

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
