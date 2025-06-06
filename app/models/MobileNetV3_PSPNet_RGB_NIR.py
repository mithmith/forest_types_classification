import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileNetV3_PSPNet_NIR(nn.Module):
    def __init__(self, num_classes: int, freeze_encoder: bool = False):
        super(MobileNetV3_PSPNet_NIR, self).__init__()
        self.num_classes = num_classes

        # MobileNetV3 Large encoder
        self.encoder = timm.create_model("mobilenetv3_large_100", pretrained=True, features_only=True)
        if freeze_encoder:
            self.freeze_rgb_layers()  # Вызываем заморозку сразу после загрузки весов
        self.modify_first_layer(in_channels=4)
        encoder_channels = self.encoder.feature_info.channels()[-1]  # Выходные каналы последнего уровня энкодера

        # Pyramid Pooling Module
        self.ppm = PyramidPoolingModule(encoder_channels, [1, 2, 3, 6])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_classes, kernel_size=1),
        )

        # Initialize weights for new layers
        self.initialize_weights()

    def modify_first_layer(self, in_channels: int):
        """Modify the first convolutional layer to accept additional input channels."""
        first_layer_name = list(self.encoder._modules.keys())[0]  # First layer name
        first_layer = getattr(self.encoder, first_layer_name)

        if isinstance(first_layer, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels,
                first_layer.out_channels,
                kernel_size=first_layer.kernel_size,
                stride=first_layer.stride,
                padding=first_layer.padding,
                bias=False,
            )
            # Initialize weights for the new channels
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = first_layer.weight  # Copy RGB weights
                nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :], mode="fan_out", nonlinearity="relu")

            # Replace the layer in the encoder
            setattr(self.encoder, first_layer_name, new_conv)
        else:
            raise ValueError(f"First layer is not a Conv2d layer, found: {type(first_layer)}")

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
        enc_out = self.encoder(x)[-1]  # Output from the last stage of the encoder

        # Pyramid Pooling Module
        ppm_out = self.ppm(enc_out)

        # Decoder
        output = self.decoder(ppm_out)
        output = F.interpolate(output, size=x.size()[2:], mode="bilinear", align_corners=False)  # Resize to input size
        return output


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(in_channels // 4),
                    nn.ReLU(inplace=True),
                )
                for _ in range(len(pool_sizes))
            ]
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * (in_channels // 4), in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        ppm_outs = [x]
        for stage in self.stages:
            pooled = stage(x)
            ppm_outs.append(F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False))
        ppm_out = torch.cat(ppm_outs, dim=1)
        return self.conv(ppm_out)
