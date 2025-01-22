import torch
import torch.nn as nn
import timm


class MobileNetV3_UNet_NIR_fMASK(nn.Module):
    def __init__(self, num_classes: int):
        super(MobileNetV3_UNet_NIR_fMASK, self).__init__()
        self.num_classes = num_classes

        # MobileNetV3 encoder
        self.encoder = timm.create_model("mobilenetv3_large_100", pretrained=True, features_only=True)
        self.freeze_rgb_layers()  # Вызываем заморозку сразу после загрузки весов
        self.modify_first_layer(in_channels=5)
        encoder_channels = self.encoder.feature_info.channels()  # Get output channels of all encoder stages

        # Decoder with skip connections
        self.dec4 = self.build_decoder_block(encoder_channels[4], encoder_channels[3])
        self.dec3 = self.build_decoder_block(encoder_channels[3] + encoder_channels[3], encoder_channels[2])
        self.dec2 = self.build_decoder_block(encoder_channels[2] + encoder_channels[2], encoder_channels[1])
        self.dec1 = self.build_decoder_block(encoder_channels[1] + encoder_channels[1], encoder_channels[0])
        self.dec0 = self.build_decoder_block(encoder_channels[0] + 16, 32)

        # Final output layer
        self.final_conv = nn.Conv2d(32, self.num_classes, kernel_size=1)

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
