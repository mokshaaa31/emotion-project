import torch
import torch.nn as nn

class CrossAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ✅ FIXED DIMENSION
        self.video_proj = nn.Linear(1024, 256)
        self.audio_proj = nn.Linear(256, 256)

        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, video_feat, audio_feat):
        v = self.video_proj(video_feat).unsqueeze(1)
        a = self.audio_proj(audio_feat).unsqueeze(1)

        fused, _ = self.attn(v, a, a)
        fused = fused.squeeze(1)

        return self.fc(fused)