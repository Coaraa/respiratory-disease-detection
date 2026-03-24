import sys
import numpy as np
import torch
import librosa
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'notebooks'))
from ConvNeXt import GPUAugmenter, RespiratoryModel

sys.path.insert(0, str(ROOT / 'app'))
from config import SR  # noqa: E402
from audio_utils import preprocess_audio, get_embedding  # noqa: E402

N_REFS = 3   # fichiers par classe pour la moyenne

CLASS_TO_FOLDER = {
    'asthma':    'asthma',
    'bronchial': 'Bronchial',
    'copd':      'copd',
    'healthy':   'healthy',
    'pneumonia': 'pneumonia',
}

# Main 
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