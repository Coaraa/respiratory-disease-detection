import numpy as np
import torch
import librosa
from scipy.signal import butter, filtfilt
from config import SR, TARGET_LEN, CLASSES


def bandpass_filter(audio, lowcut=100, highcut=2000, sr=SR, order=4):
    nyq = sr / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, audio).astype(np.float32)


def pad_or_crop(audio, target_len=TARGET_LEN):
    if len(audio) < target_len:
        pad = target_len - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    else:
        audio = audio[:target_len]
    return audio


def preprocess_audio(audio):
    """Pipeline : trim → bandpass 100-2000 Hz → normalise → pad/crop 6s."""
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = bandpass_filter(audio)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return pad_or_crop(audio)


def get_embedding(audio_array, model, augmenter, device):
    tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        spec     = augmenter(tensor, augment=False)
        features = model.cnn(spec)
        return features.squeeze().cpu().numpy()


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def compare_to_references(embedding, refs):
    sims = {cls: cosine_similarity(embedding, refs[cls]) for cls in CLASSES if cls in refs}
    lo, hi = min(sims.values()), max(sims.values())
    span = hi - lo if hi > lo else 1.0
    return {cls: (v - lo) / span for cls, v in sims.items()}
