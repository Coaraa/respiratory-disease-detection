"""
Génère les embeddings de référence (vecteurs 768-dim ConvNeXt) pour 3 fichiers
par classe, puis calcule la moyenne par classe et sauvegarde dans
models/reference_embeddings.npz.

Usage :
    python scripts/generate_references.py
"""
import sys
import numpy as np
import torch
import librosa
from pathlib import Path
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'notebooks'))
from ConvNeXt import GPUAugmenter, RespiratoryModel

# ── Constantes ──────────────────────────────────────────────────────────────
SR         = 22050
TARGET_LEN = SR * 6
N_REFS     = 3   # fichiers par classe pour la moyenne

CLASS_TO_FOLDER = {
    'asthma':    'asthma',
    'bronchial': 'Bronchial',
    'copd':      'copd',
    'healthy':   'healthy',
    'pneumonia': 'pneumonia',
}

# ── Preprocessing ────────────────────────────────────────────────────────────
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
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = bandpass_filter(audio)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return pad_or_crop(audio)

# ── Extraction embedding (avant la tête de classification) ──────────────────
def get_embedding(audio_array, model, augmenter, device):
    tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        spec     = augmenter(tensor, augment=False)
        features = model.cnn(spec)                     # (1, 768, 1, 1)
        return features.squeeze().cpu().numpy()        # (768,)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    pth_path  = ROOT / 'models' / 'best_model_convnext.pth'

    print(f"Chargement du modèle ({device})...")
    model     = RespiratoryModel(num_classes=5, weights=None).to(device)
    augmenter = GPUAugmenter().to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    augmenter.eval()

    embeddings = {}
    for cls, folder in CLASS_TO_FOLDER.items():
        folder_path = ROOT / 'data' / folder
        wav_files   = sorted(folder_path.glob('*.wav'))[:N_REFS]
        cls_embs    = []
        for wav_path in wav_files:
            audio, _ = librosa.load(wav_path, sr=SR)
            audio    = preprocess_audio(audio)
            emb      = get_embedding(audio, model, augmenter, device)
            cls_embs.append(emb)
        embeddings[cls] = np.mean(cls_embs, axis=0)
        print(f"  {cls:12s}: {len(cls_embs)} fichier(s) traité(s)")

    out_path = ROOT / 'models' / 'reference_embeddings.npz'
    np.savez(out_path, **embeddings)
    print(f"\nEmbeddings sauvegardés → {out_path}")

if __name__ == '__main__':
    main()