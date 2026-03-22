import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils.video_utils import get_frames
from utils.audio_utils import extract_audio, extract_audio_features

from models.model import VideoTransformer
from models.audio_model import AudioEncoder
from models.fusion_model import CrossAttentionModel

torch.manual_seed(42)


# 📦 DATASET
class MultiModalDataset(Dataset):
    def __init__(self, data_dir):
        self.files = []
        self.labels = []

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".mp4"):
                    path = os.path.join(root, file)

                    emotion_code = int(file.split("-")[2])

                    if emotion_code == 1:
                        label = 3  # neutral
                    elif emotion_code == 3:
                        label = 0  # happy
                    elif emotion_code == 4:
                        label = 1  # sad
                    elif emotion_code == 5:
                        label = 2  # angry
                    else:
                        continue

                    self.files.append(path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        video_path = self.files[idx]
        label = self.labels[idx]

        # 🎥 VIDEO (MULTI-FRAMES)
        frames = get_frames(video_path)

        processed_frames = []
        for frame in frames:
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            frame = torch.tensor(frame).float()
            processed_frames.append(frame)

        if len(processed_frames) == 0:
            processed_frames.append(torch.zeros((3, 224, 224)))

        frames_tensor = torch.stack(processed_frames)

        # 🎧 AUDIO
        audio_path = extract_audio(video_path)
        audio_feat = extract_audio_features(audio_path)
        audio_feat = torch.tensor(audio_feat).float()

        return frames_tensor, audio_feat, label


# 🚀 LOAD DATA
dataset = MultiModalDataset("data")
loader = DataLoader(dataset, batch_size=1, shuffle=True)


# 🧠 MODELS
video_model = VideoTransformer()
audio_model = AudioEncoder()
fusion_model = CrossAttentionModel()
# 🔥 ADD HERE (FREEZE VIDEO MODEL)
for param in video_model.parameters():
    param.requires_grad = False

# 🔧 OPTIMIZER
params = list(video_model.parameters()) + \
         list(audio_model.parameters()) + \
         list(fusion_model.parameters())

optimizer = torch.optim.Adam(params, lr=0.0005)

# 🔥 BALANCED LOSS (IMPORTANT FIX)
weights = torch.tensor([1.2, 1.2, 1.2, 1.0])
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)


# 🚀 TRAIN LOOP
for epoch in range(8):
    total_loss = 0

    for frames, audio_feat, label in loader:

        video_outputs = []

        for i in range(frames.shape[1]):
            frame = frames[:, i, :, :, :]
            with torch.no_grad():
                 video_feat = video_model(frame)
            video_outputs.append(video_feat)

        video_feat = torch.mean(torch.stack(video_outputs), dim=0)

        audio_feat = audio_model(audio_feat)

        output = fusion_model(video_feat, audio_feat)

        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


# 💾 SAVE
torch.save(video_model.state_dict(), "video_model.pth")
torch.save(audio_model.state_dict(), "audio_model.pth")
torch.save(fusion_model.state_dict(), "fusion_model.pth")

print("✅ Training complete (balanced + multi-frame)")