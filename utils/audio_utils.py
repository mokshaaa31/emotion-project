import librosa
import numpy as np
import subprocess

def extract_audio(video_path, out_path="temp.wav"):
    command = ["ffmpeg", "-y", "-i", video_path, out_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc