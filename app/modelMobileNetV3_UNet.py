import timm
import torch
import torch.nn as nn


class MobileNetV3_UNet(nn.Module):
    def __init__(self, num_classes: int):
        super(MobileNetV3_UNet, self).__init__()
        self.num_classes = num_classes

        # MobileNetV3 encoder
        self.encoder = timm.create_model("mobilenetv3_large_100", pretrained=True, features_only=True)
        self.freeze_rgb_layers()  # Вызываем заморозку сразу после загрузки весов
        encoder_channels = self.encoder.feature_info.channels()  # Get output channels of all encoder stages

        # Decoder with skip connections
        self.dec4 = self.build_decoder_block(encoder_channels[4], encoder_channels[3])
        self.dec3 = self.build_decoder_block(encoder_channels[3] + encoder_channels[3], encoder_channels[2])
        self.dec2 = self.build_decoder_block(encoder_channels[2] + encoder_channels[2], encoder_channels[1])
        self.dec1 = self.build_decoder_block(encoder_channels[1] + encoder_channels[1], encoder_channels[0])
        self.dec0 = self.build_decoder_block(encoder_channels[0] + 16, 32)

        # Final output layer
        self.final_conv = nn.Conv2d(32, self.num_classes, kernel_size=1)
        self.initialize_weights()

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
        # Заморозка всех слоев энкодера для RGB
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

    def forward(self, x: torch.Tensor):
        # Encoder
        enc_outs = self.encoder(x)  # Get features from encoder levels

        # Decoder with skip connections
        dec4_out = self.dec4(enc_outs[4])  # 16x16 -> 32x32
        dec3_out = self.dec3(torch.cat([dec4_out, enc_outs[3]], dim=1))  # 32x32 -> 64x64
        dec2_out = self.dec2(torch.cat([dec3_out, enc_outs[2]], dim=1))  # 64x64 -> 128x128
        dec1_out = self.dec1(torch.cat([dec2_out, enc_outs[1]], dim=1))  # 128x128 -> 256x256
        output = self.dec0(torch.cat([dec1_out, enc_outs[0]], dim=1))  # 256x256 -> 512x512
        return self.final_conv(output)
