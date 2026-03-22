import torch
import torch.nn as nn
import timm

class VideoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # ✅ LIGHTWEIGHT MODEL (FAST)
        self.model = timm.create_model("mobilenetv3_small_100", pretrained=True)
        self.model.classifier = nn.Identity()

    def forward(self, x):
        return self.model(x)  # (B, ~576)