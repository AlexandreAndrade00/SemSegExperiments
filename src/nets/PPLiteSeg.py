from typing import Optional

import torch.nn.functional as F
from torch import nn

from modules import SPPM, ConvBNAct, UAFM_SpAtten
from nets import STDCNet


class PPLiteSeg(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module],
        ppm: Optional[nn.Module],
        attention_fusion: str = "spatial",
    ):
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
        self.seg_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(64, num_classes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        # (H, W)
        image_xy = x.shape[:2]

        # features from encoder's stages, the encoder must output 5 features
        encoder_features = self.backbone(x)

        # input to the ppm the highest level feature from encoder
        ppm_out = self.ppm(encoder_features[4])

        if self.attention_fusion == "spatial":
            decoder_1 = UAFM_SpAtten(self.backbone.feat_channels[-2], 128, 96)
        else:
            raise NotImplementedError()

        decoder_1_out = decoder_1(encoder_features[3], ppm_out)

        if self.attention_fusion == "spatial":
            decoder_2 = UAFM_SpAtten(self.backbone.feat_channels[-3], 96, 64)
        else:
            raise NotImplementedError()

        decoder_2_out = decoder_2(encoder_features[2], decoder_1_out)

        seg_output = self.seg_head(decoder_2_out)

        resized_seg_output = F.interpolate(
            seg_output, image_xy, mode="bilinear", align_corners=False
        )

        return resized_seg_output
