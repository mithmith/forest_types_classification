import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet50_PSPNet(nn.Module):
    def __init__(self, num_classes: int, pool_sizes=(1, 2, 3, 6)):
        super(ResNet50_PSPNet, self).__init__()
        self.num_classes = num_classes
        self.pool_sizes = pool_sizes

        # ResNet50 encoder
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.freeze_rgb_layers()  # Вызываем заморозку сразу после загрузки весов

        # Save intermediate features
        self.enc1 = nn.Sequential(*list(self.encoder.children())[:3])  # Conv1 + BN + ReLU
        self.enc2 = nn.Sequential(*list(self.encoder.children())[3:5])  # MaxPool + Layer1
        self.enc3 = self.encoder.layer2
        self.enc4 = self.encoder.layer3
        self.enc5 = self.encoder.layer4

        # Pyramid Pooling Module
        self.ppm = self.build_pyramid_pooling_module(2048, 512, pool_sizes)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(2048 + len(pool_sizes) * 512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Final output layer
        self.final_conv = nn.Conv2d(256, self.num_classes, kernel_size=1)

        # Initialize weights for new layers
        self.initialize_weights()

    def build_pyramid_pooling_module(self, in_channels, out_channels, pool_sizes):
        """Create the Pyramid Pooling Module."""
        ppm = []
        for pool_size in pool_sizes:
            ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_size),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        return nn.ModuleList(ppm)

    def freeze_rgb_layers(self):
        # Заморозка всех слоев энкодера для RGB из ResNet50
        for param in self.encoder.parameters():
            param.requires_grad = False

    def initialize_weights(self):
        """Initialize the weights of the newly added layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)  # 512x512 -> 256x256
        enc2_out = self.enc2(enc1_out)  # 256x256 -> 128x128
        enc3_out = self.enc3(enc2_out)  # 128x128 -> 64x64
        enc4_out = self.enc4(enc3_out)  # 64x64 -> 32x32
        enc5_out = self.enc5(enc4_out)  # 32x32 -> 16x16

        # Pyramid Pooling Module
        ppm_outs = [enc5_out]
        for ppm_layer in self.ppm:
            ppm_out = ppm_layer(enc5_out)
            ppm_out = F.interpolate(ppm_out, size=enc5_out.shape[2:], mode="bilinear", align_corners=False)
            ppm_outs.append(ppm_out)
        ppm_out = torch.cat(ppm_outs, dim=1)

        # Decoder
        decoder_out = self.decoder(ppm_out)
        decoder_out = F.interpolate(decoder_out, size=x.shape[2:], mode="bilinear", align_corners=False)

        # Final output layer
        output = self.final_conv(decoder_out)
        return output
