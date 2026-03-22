import streamlit as st
import tempfile
import torch
import numpy as np
import cv2
import os
import urllib.request

from utils.video_utils import get_frames
from utils.audio_utils import extract_audio, extract_audio_features

from models.model import VideoTransformer
from models.audio_model import AudioEncoder
from models.fusion_model import CrossAttentionModel

# -------------------------------
# ⚙️ PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Emotion Recognition", layout="centered")

st.title("🎭 Multimodal Emotion Recognition")
st.write("Upload a video to detect emotion using Audio + Video")

# -------------------------------
# 🔽 DOWNLOAD MODELS (RUN ONCE)
# -------------------------------
@st.cache_resource
def download_models():
    def download(url, filename):
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)

    download("https://drive.google.com/uc?id=1SMZL7V-nnaOItT-CvI2rm_Oyeg5zkS1i", "video_model.pth")
    download("https://drive.google.com/uc?id=18grYKARGAt0qRz1qNui3tnWyzGCYdiu_", "audio_model.pth")
    download("https://drive.google.com/uc?id=1RSPfbQToHUzRDOrlQse7kNXQ5TAGucAm", "fusion_model.pth")

download_models()

# -------------------------------
# ⚡ LOAD MODELS (CACHED)
# -------------------------------
@st.cache_resource
def load_models():
    video_model = VideoTransformer()
    audio_model = AudioEncoder()
    fusion_model = CrossAttentionModel()

    video_model.load_state_dict(torch.load("video_model.pth", map_location="cpu"))
    audio_model.load_state_dict(torch.load("audio_model.pth", map_location="cpu"))
    fusion_model.load_state_dict(torch.load("fusion_model.pth", map_location="cpu"))

    video_model.eval()
    audio_model.eval()
    fusion_model.eval()

    return video_model, audio_model, fusion_model


video_model, audio_model, fusion_model = load_models()

st.success("⚡ Models loaded. Ready for prediction!")

# -------------------------------
# 🎥 FILE UPLOAD
# -------------------------------
video = st.file_uploader("Upload Video", type=["mp4", "mov"])

# -------------------------------
# 🚀 PROCESS VIDEO
# -------------------------------
if video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    st.video(video)

    with st.spinner("🔍 Analyzing emotion..."):

        # 🎥 VIDEO PROCESSING (LIMIT FRAMES FOR SPEED)
        frames = get_frames(tfile.name, max_frames=5)
        video_features = []

        for frame in frames:
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            frame = torch.tensor(frame).float().unsqueeze(0)

            with torch.no_grad():
                feat = video_model(frame)

            video_features.append(feat)

        video_feat = torch.mean(torch.stack(video_features), dim=0)

        # 🎧 AUDIO PROCESSING
        audio_path = extract_audio(tfile.name)
        audio_feat_np = extract_audio_features(audio_path)

        audio_feat = torch.tensor(audio_feat_np).float().unsqueeze(0)

        with torch.no_grad():
            audio_feat = audio_model(audio_feat)

        # 🔗 FUSION
        with torch.no_grad():
            output = fusion_model(video_feat, audio_feat)

        # ✅ STABLE PREDICTION
        probs = torch.softmax(output, dim=1).numpy()[0]
        pred = int(np.argmax(probs))

    # -------------------------------
    # 🎯 OUTPUT
    # -------------------------------
    labels = ["😊 Happy", "😢 Sad", "😠 Angry", "😐 Neutral"]

    st.success(f"🎯 Detected Emotion: {labels[pred]}")