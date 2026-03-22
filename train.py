import os
import torch
from torch.utils.data import Dataset, DataLoader
from models.model import TransformerModel
from utils.video_utils import get_frames

class EmotionDataset(Dataset):
    def __init__(self, data_dir):
        self.files = []
        self.labels = []

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".mp4"):
                    path = os.path.join(root, file)

                    # Extract emotion from filename
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
        import cv2
        import numpy as np

        video_path = self.files[idx]
        label = self.labels[idx]

        frames = get_frames(video_path)

        if len(frames) == 0:
            frame = np.zeros((224, 224, 3))
        else:
            frame = frames[0]

        # Preprocess
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.transpose(frame, (2, 0, 1))

        frame = torch.tensor(frame).float()

        return frame, label


# 🔹 Load dataset
dataset = EmotionDataset("data")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 🔹 Transformer model
model = TransformerModel()

# 🔹 Optimizer & loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # smaller LR for transformer
loss_fn = torch.nn.CrossEntropyLoss()

# 🔹 Training loop
for epoch in range(5):
    total_loss = 0

    for x, y in loader:
        output = model(x)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# 🔹 Save model
torch.save(model.state_dict(), "model.pth")

print("✅ Transformer training complete. Model saved as model.pth")