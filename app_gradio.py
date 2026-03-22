import gradio as gr
import tempfile
import torch
import numpy as np
import cv2

from utils.video_utils import get_frames
from utils.audio_utils import extract_audio, extract_audio_features

from models.model import VideoTransformer
from models.audio_model import AudioEncoder
from models.fusion_model import CrossAttentionModel

# LOAD MODELS
video_model = VideoTransformer()
audio_model = AudioEncoder()
fusion_model = CrossAttentionModel()

video_model.load_state_dict(torch.load("video_model.pth", map_location="cpu"))
audio_model.load_state_dict(torch.load("audio_model.pth", map_location="cpu"))
fusion_model.load_state_dict(torch.load("fusion_model.pth", map_location="cpu"))

video_model.eval()
audio_model.eval()
fusion_model.eval()


def predict(video):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video)

    # VIDEO
    frames = get_frames(tfile.name)
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

    # AUDIO
    audio_path = extract_audio(tfile.name)
    audio_feat_np = extract_audio_features(audio_path)
    audio_feat = torch.tensor(audio_feat_np).float().unsqueeze(0)

    with torch.no_grad():
        audio_feat = audio_model(audio_feat)

    # FUSION
    with torch.no_grad():
        output = fusion_model(video_feat, audio_feat)

    pred = torch.argmax(output).item()

    labels = ["Happy 😊", "Sad 😢", "Angry 😠", "Neutral 😐"]

    return labels[pred]


iface = gr.Interface(
    fn=predict,
    inputs=gr.Video(),
    outputs="text",
    title="🎭 Emotion Recognition (Audio + Video)",
    description="Upload a video to detect emotion"
)

iface.launch()