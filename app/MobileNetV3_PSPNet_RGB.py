import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MobileNetV3_PSPNet(nn.Module):
    def __init__(self, num_classes: int):
        super(MobileNetV3_PSPNet, self).__init__()
        self.num_classes = num_classes

        # MobileNetV3 Large encoder
        self.encoder = timm.create_model("mobilenetv3_large_100", pretrained=True, features_only=True)
        self.freeze_rgb_layers()  # Вызываем заморозку сразу после загрузки весов
        encoder_channels = self.encoder.feature_info.channels()[-1]  # Выходные каналы последнего уровня энкодера

        # Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(encoder_channels, [1, 2, 3, 6])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_channels * (encoder_channels // 4), 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_classes, kernel_size=1)
        )

        # Initialize weights for new layers
        self.initialize_weights()

    def freeze_rgb_layers(self):
        # Заморозка всех слоев энкодера для RGB из ResNet50
        for param in self.encoder.parameters():
            param.requires_grad = False

    def initialize_weights(self):
        """Initialize the weights of the newly added layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        enc_out = self.encoder(x)[-1]  # Output from the last stage of the encoder

        # Pyramid Pooling Module
        ppm_out = self.ppm(enc_out)

        # Decoder
        output = self.decoder(ppm_out)
        output = F.interpolate(output, size=x.size()[2:], mode='bilinear', align_corners=False)  # Resize to input size
        return output


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            ) for pool_size in pool_sizes
        ])
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * (in_channels // 4), in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        ppm_outs = [x]
        for stage in self.stages:
            pooled = stage(x)
            ppm_outs.append(F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_outs, dim=1)
        return self.conv(ppm_out)
