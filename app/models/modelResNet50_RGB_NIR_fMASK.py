import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict
from typing import OrderedDict as OD
from typing import Union

import torch
import torch.nn as nn
import torchvision.models as models



class ResNet50_UNet_NIR_fMASK(nn.Module):
    DECODER_REGEX = re.compile(r"dec(\d)\.(\d+)\.(.*)")

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 5,
        encoder_weights: bool = False,  # True -> ImageNet-претрен, False -> random
        freeze_encoder: bool = False,  # True -> заморозить энкодер
        dropout_p: float = 0.1,
    ):
        super(ResNet50_UNet_NIR_fMASK, self).__init__()
        self.num_classes = num_classes

        # Single ResNet encoder
        resnet_weights = models.ResNet50_Weights.IMAGENET1K_V2 if encoder_weights else None
        self.encoder = models.resnet50(weights=resnet_weights)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False  # Вызываем заморозку сразу после загрузки весов

        self.encoder.conv1 = self._make_first_conv(self.encoder.conv1, in_channels)

        # Save intermediate features for skip connections
        self.enc1 = nn.Sequential(*list(self.encoder.children())[:3])  # Conv1 + BN + ReLU
        self.enc2 = nn.Sequential(*list(self.encoder.children())[3:5])  # MaxPool + Layer1
        self.enc3 = self.encoder.layer2
        self.enc4 = self.encoder.layer3
        self.enc5 = self.encoder.layer4

        # Decoder with skip connections
        self.dec4 = self._decoder_block(2048, 1024, p=dropout_p)
        self.dec3 = self._decoder_block(1024 + 1024, 512, p=dropout_p)
        self.dec2 = self._decoder_block(512 + 512, 256, p=dropout_p)
        self.dec1 = self._decoder_block(256 + 256, 128, p=dropout_p)
        self.dec0 = self._decoder_block(128 + 64, 64, p=dropout_p)

        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    @staticmethod
    def _make_first_conv(old: nn.Conv2d, in_ch: int) -> nn.Conv2d:
        """Создаём новую conv1 и копируем первые 3 канала из старой."""
        new = nn.Conv2d(
            in_ch,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        with torch.no_grad():
            c = min(3, in_ch)
            new.weight[:, :c] = old.weight[:, :c]
            if in_ch > 3:
                nn.init.kaiming_normal_(new.weight[:, 3:], mode="fan_out", nonlinearity="relu")
        return new

    @staticmethod
    def _decoder_block(in_ch: int, out_ch: int, p: float = 0.1) -> nn.Sequential:
        """Build a single block of the decoder."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @classmethod
    def _shift_decoder_keys(cls, state: Dict[str, torch.Tensor]) -> OD[str, torch.Tensor]:
        """decX.3-5 → decX.4-6 (для старых чек-пойнтов без Dropout)."""
        new_state: OD[str, torch.Tensor] = OrderedDict()
        for k, v in state.items():
            m = cls.DECODER_REGEX.match(k)
            if m:
                blk, idx, rest = m.groups()
                if int(idx) >= 3:
                    k = f"dec{blk}.{int(idx)+1}.{rest}"
            new_state[k] = v
        return new_state

    @staticmethod
    def _expand_conv_weight(w: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Расширяем conv-вес по размеру входных каналов."""
        if w.shape == target_shape:
            return w

        if w.shape[2:] != target_shape[2:]:
            raise ValueError("conv1 kernel size mismatch")

        new_w = torch.zeros(target_shape, dtype=w.dtype, device=w.device)
        c = min(w.shape[1], target_shape[1])
        new_w[:, :c] = w[:, :c]
        nn.init.kaiming_normal_(new_w[:, c:], mode="fan_out", nonlinearity="relu")
        return new_w

    @torch.no_grad()
    def load_weights_adaptively(
        self,
        checkpoint: Union[str, Path],
        device: str = "cpu",
    ) -> None:
        """
        Пытаемся загрузить «новый» чек-пойнт. Если не получается —
        конвертируем «старый» формат.
        """
        raw = torch.load(checkpoint, map_location=device)
        state: Dict[str, torch.Tensor] = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

        # --- 1. прямой импорт ---------------------------------------------
        try:
            self.load_state_dict(state, strict=True)
            return
        except RuntimeError:
            pass

        # --- 2. починка старого декодера ----------------------------------
        state = self._shift_decoder_keys(state)

        # --- 3. conv1: 4 → 5 каналов и т. д. ------------------------------
        conv_key = "encoder.conv1.weight"
        if conv_key in state:
            target_shape = self.state_dict()[conv_key].shape
            try:
                state[conv_key] = self._expand_conv_weight(state[conv_key], target_shape)
            except ValueError:
                del state[conv_key]

        # --- 4. выбрасываем shape-mismatch --------------------------------
        clean_state: OD[str, torch.Tensor] = OrderedDict()
        ref_state = self.state_dict()
        for k, v in state.items():
            if k in ref_state and v.shape == ref_state[k].shape:
                clean_state[k] = v

        self.load_state_dict(clean_state, strict=False)
        miss = len(ref_state) - len(clean_state)
        skip = len(state) - len(clean_state)
        print(f"✓ legacy checkpoint loaded (used {len(clean_state)}, miss {miss}, skip {skip})")

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)  # 512x512 -> 256x256
        enc2_out: torch.Tensor = self.enc2(enc1_out)  # 256x256 -> 128x128
        enc3_out: torch.Tensor = self.enc3(enc2_out)  # 128x128 -> 64x64
        enc4_out: torch.Tensor = self.enc4(enc3_out)  # 64x64 -> 32x32
        enc5_out: torch.Tensor = self.enc5(enc4_out)  # 32x32 -> 16x16

        # Decoder with skip connections
        dec4_out = self.dec4(enc5_out)  # 16x16 -> 32x32
        dec3_out = self.dec3(torch.cat([dec4_out, enc4_out], dim=1))  # 32x32 -> 64x64
        dec2_out = self.dec2(torch.cat([dec3_out, enc3_out], dim=1))  # 64x64 -> 128x128
        dec1_out = self.dec1(torch.cat([dec2_out, enc2_out], dim=1))  # 128x128 -> 256x256
        output = self.dec0(torch.cat([dec1_out, enc1_out], dim=1))  # 256x256 -> 512x512
        return self.final_conv(output)
