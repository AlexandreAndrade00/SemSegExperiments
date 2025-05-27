from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from src.modules import SPPM, ConvBNAct, UAFM_SpAtten

from .STDCNet import STDCNet


class PPLiteSeg(nn.Module):
    name: str = "pplite-seg"

    def __init__(
        self,
        num_classes_train: int,
        num_classes_pretrain: int,
        device: torch.device,
        backbone: Optional[nn.Module] = None,
        ppm: Optional[nn.Module] = None,
        attention_fusion: str = "spatial",
    ):
        super().__init__()

        self.num_classes_train = num_classes_train
        self.num_classes_pretrain = num_classes_pretrain
        self.device = device
        self.mode: str = "normal"

        self.backbone = backbone if backbone is not None else STDCNet()
        self.attention_fusion = attention_fusion
        self.ppm = (
            ppm
            if ppm is not None
            else SPPM(
                in_channels=self.backbone.feat_channels[-1],
                inter_channels=128,
                out_channels=128,
            )
        )

        if self.attention_fusion == "spatial":
            self.decoder_1 = UAFM_SpAtten(self.backbone.feat_channels[-2], 128, 96)
        else:
            raise NotImplementedError()

        if self.attention_fusion == "spatial":
            self.decoder_2 = UAFM_SpAtten(self.backbone.feat_channels[-3], 96, 64)
        else:
            raise NotImplementedError()

        self.seg_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(64, num_classes_train, kernel_size=1, bias=False),
        )

        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(64, num_classes_pretrain, bias=False),
        )

    def normal_train(self) -> None:
        self.mode = "normal"

    def pre_train(self) -> None:
        self.mode = "pretrain"

    def forward(self, x: Tensor) -> Tensor:
        # (H, W)
        image_xy = x.shape[2:]

        # features from encoder's stages, the encoder must output 5 features
        encoder_features = self.backbone(x)

        # input to the ppm the highest level feature from encoder
        ppm_out = self.ppm(encoder_features[4])

        decoder_1_out = self.decoder_1(encoder_features[3], ppm_out)

        decoder_2_out = self.decoder_2(encoder_features[2], decoder_1_out)

        if self.mode == "normal":
            seg_output = self.seg_head(decoder_2_out)

            resized_seg_output: Tensor = F.interpolate(
                seg_output, image_xy, mode="bilinear", align_corners=False
            )

            return resized_seg_output
        elif self.mode == "pretrain":
            out: Tensor = self.classification_head(decoder_2_out)

            return out
        else:
            raise ValueError(self.mode)
