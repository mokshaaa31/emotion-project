import os
import torch
import cv2
import numpy as np

from utils.video_utils import get_frames
from utils.audio_utils import extract_audio, extract_audio_features
from models.model import VideoTransformer
from models.audio_model import AudioEncoder

video_model = VideoTransformer()
audio_model = AudioEncoder()

video_model.eval()
audio_model.eval()

features = []
labels = []

data_dir = "data"

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".mp4"):
            path = os.path.join(root, file)

            emotion_code = int(file.split("-")[2])

            if emotion_code == 1:
                label = 3
            elif emotion_code == 3:
                label = 0
            elif emotion_code == 4:
                label = 1
            elif emotion_code == 5:
                label = 2
            else:
                continue

            # 🎥 VIDEO
            frames = get_frames(path)
            video_outputs = []

            for frame in frames:
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frame = np.transpose(frame, (2, 0, 1))
                frame = torch.tensor(frame).float().unsqueeze(0)

                with torch.no_grad():
                    feat = video_model(frame)

                video_outputs.append(feat)

            video_feat = torch.mean(torch.stack(video_outputs), dim=0)

            # 🎧 AUDIO
            audio_path = extract_audio(path)
            audio_feat_np = extract_audio_features(audio_path)
            audio_feat = torch.tensor(audio_feat_np).float().unsqueeze(0)

            with torch.no_grad():
                audio_feat = audio_model(audio_feat)

            # 🔗 SAVE
            features.append(torch.cat([video_feat.squeeze(), audio_feat.squeeze()]))
            labels.append(label)

# SAVE
torch.save((features, labels), "features.pt")

print("✅ Features extracted and saved")