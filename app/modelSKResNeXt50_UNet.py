import timm
import torch
import torch.nn as nn


class SKResNeXt50_UNet(nn.Module):
    def __init__(self, num_classes: int):
        super(SKResNeXt50_UNet, self).__init__()
        self.num_classes = num_classes

        # SK-ResNeXt50 encoder (timm library)
        self.encoder = timm.create_model("skresnext50_32x4d", pretrained=True, features_only=True)
        self.freeze_rgb_layers()  # Вызываем заморозку сразу после загрузки весов
        encoder_channels = self.encoder.feature_info.channels()  # Получаем список каналов для всех слоев

        # Decoder with skip connections
        self.dec4 = self.build_decoder_block(encoder_channels[4], encoder_channels[3])
        self.dec3 = self.build_decoder_block(encoder_channels[3] + encoder_channels[3], encoder_channels[2])
        self.dec2 = self.build_decoder_block(encoder_channels[2] + encoder_channels[2], encoder_channels[1])
        self.dec1 = self.build_decoder_block(encoder_channels[1] + encoder_channels[1], encoder_channels[0])
        self.dec0 = self.build_decoder_block(encoder_channels[0] + 64, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, self.num_classes, kernel_size=1)
        self.initialize_weights()

    def build_decoder_block(self, in_channels, out_channels):
        """Build a single block of the decoder."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            # nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def freeze_rgb_layers(self):
        # Заморозка всех слоев энкодера для RGB
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()  # Фиксируем веса BatchNorm

    def initialize_weights(self):
        """Initialize the weights of the newly added layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        # Encoder
        enc_outs = self.encoder(x)  # Получаем фичи с уровней энкодера
        enc_outs = [torch.clamp(out, min=0, max=1e5) for out in enc_outs]
        # Проверяем слои энкодера
        for i, enc in enumerate(enc_outs):
            print(f"enc_outs[{i}] min: {enc.min().item()}, max: {enc.max().item()}")

        # Decoder with skip connections
        dec4_out = self.dec4(enc_outs[4])  # 16x16 -> 32x32
        print(f"dec4_out min: {dec4_out.min().item()}, max: {dec4_out.max().item()}")

        dec3_out = self.dec3(torch.cat([dec4_out, enc_outs[3]], dim=1))  # 32x32 -> 64x64
        print(f"dec3_out min: {dec3_out.min().item()}, max: {dec3_out.max().item()}")

        dec2_out = self.dec2(torch.cat([dec3_out, enc_outs[2]], dim=1))  # 64x64 -> 128x128
        print(f"dec2_out min: {dec2_out.min().item()}, max: {dec2_out.max().item()}")

        dec1_out = self.dec1(torch.cat([dec2_out, enc_outs[1]], dim=1))  # 128x128 -> 256x256
        print(f"dec1_out min: {dec1_out.min().item()}, max: {dec1_out.max().item()}")

        output = self.dec0(torch.cat([dec1_out, enc_outs[0]], dim=1))  # 256x256 -> 512x512
        print(f"Output перед финальным conv: min={output.min().item()}, max={output.max().item()}")

        decoder_output = torch.clamp(output, min=-1e5, max=1e5)
        decoder_output = decoder_output / (torch.max(decoder_output) + 1e-6)
        output = self.final_conv(decoder_output)
        print(f"Output после final_conv: min={output.min().item()}, max={output.max().item()}")
        return output
