import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn  # noqa: F401 — utilisé par ConvNeXt.py importé ci-dessous
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import streamlit as st
from pathlib import Path
from scipy.signal import butter, filtfilt
from dotenv import load_dotenv

# Imports depuis notebooks/ConvNeXt.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'notebooks'))
from ConvNeXt import GPUAugmenter, RespiratoryModel
sys.path.insert(0, str(Path(__file__).parent.parent))
from snowflake_conn import get_snowflake_connection

load_dotenv(Path(__file__).parent.parent.parent / '.env')

# ── Constantes audio (identiques à data_preprocess.ipynb) ──────────────────
SR         = 22050
TARGET_LEN = SR * 6  # 132 300 samples — 6 secondes
CLASSES    = ['asthma', 'bronchial', 'copd', 'healthy', 'pneumonia']
CLASS_MAP  = {
    'asthma':    'Asthme',
    'bronchial': 'Bronchite',
    'copd':      'BPCO',
    'healthy':   'Sain',
    'pneumonia': 'Pneumonie',
}
PHARMACIES = [
    "PHARM_BORDEAUX_001", "PHARM_PARIS_002", "PHARM_LYON_003",
    "PHARM_MARSEILLE_004", "PHARM_TOULOUSE_005",
]
CLINICAL_RECS = {
    'asthma':    ("⚠️", "warning", "Suspicion d'asthme (sifflements détectés). Consultation médicale recommandée pour confirmation et prescription éventuelle de bronchodilatateurs."),
    'bronchial': ("⚠️", "warning", "Suspicion de syndrome bronchite. Bilan ORL/pulmonaire conseillé."),
    'copd':      ("🚨", "error",   "Suspicion de BPCO. Examen spirométrique urgent recommandé."),
    'healthy':   ("✅", "success", "Murmure vésiculaire régulier. Aucune anomalie respiratoire majeure détectée."),
    'pneumonia': ("🚨", "error",   "Suspicion de pneumonie. Consultation médicale urgente et radiographie pulmonaire recommandées."),
}

# ── Preprocessing (identique à data_preprocess.ipynb) ──────────────────────
def bandpass_filter(audio, lowcut=100, highcut=2000, sr=SR, order=4):
    nyquist = sr / 2
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    return filtfilt(b, a, audio).astype(np.float32)

def pad_or_crop(audio, target_len=TARGET_LEN):
    if len(audio) < target_len:
        pad_total = target_len - len(audio)
        pad_left  = pad_total // 2
        audio = np.pad(audio, (pad_left, pad_total - pad_left))
    else:
        audio = audio[:target_len]
    return audio

def preprocess_audio(audio):
    """Pipeline : trim → bandpass 100-2000 Hz → normalise → pad/crop 6s."""
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = bandpass_filter(audio)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return pad_or_crop(audio)

# ── Chargement modèle (mis en cache) ───────────────────────────────────────
@st.cache_resource
def load_model():
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    pth_path  = Path(__file__).parent.parent.parent / 'models' / 'best_model_convnext.pth'
    if not pth_path.exists():
        st.error(f"Modèle introuvable : {pth_path}")
        return None, None, device
    model     = RespiratoryModel(num_classes=5, weights=None).to(device)
    augmenter = GPUAugmenter().to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    augmenter.eval()
    return model, augmenter, device

# ── Inférence ──────────────────────────────────────────────────────────────
def predict(audio_array, model, augmenter, device):
    tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        spec    = augmenter(tensor, augment=False)
        outputs = model(spec)
        probs   = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    return dict(zip(CLASSES, probs.tolist()))

def get_embedding(audio_array, model, augmenter, device):
    """Vecteur 768-dim ConvNeXt (avant la tête de classification)."""
    tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        spec     = augmenter(tensor, augment=False)
        features = model.cnn(spec)                   # (1, 768, 1, 1)
        return features.squeeze().cpu().numpy()      # (768,)

# ── Embeddings de référence ─────────────────────────────────────────────────
@st.cache_resource
def load_reference_embeddings():
    npz_path = Path(__file__).parent.parent.parent / 'models' / 'reference_embeddings.npz'
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    return {cls: data[cls] for cls in CLASSES if cls in data}

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def compare_to_references(embedding, refs):
    """Retourne un dict {classe: similarité 0-100%} normalisé."""
    sims  = {cls: cosine_similarity(embedding, refs[cls]) for cls in CLASSES if cls in refs}
    # Normalise [min,max] → [0,1] pour affichage lisible
    lo, hi = min(sims.values()), max(sims.values())
    span  = hi - lo if hi > lo else 1.0
    return {cls: (v - lo) / span for cls, v in sims.items()}

def get_sf_connection():
    """Récupère la connexion partagée via le TOTP stocké dans l'URL."""
    totp = st.query_params.get("_sf")
    if totp:
        return get_snowflake_connection(totp)
    return None

# ── Insert Snowflake ────────────────────────────────────────────────────────
def insert_prediction(pharmacie_id, classe_predite, probabilites_dict, confiance):
    conn = get_sf_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        import uuid
        cursor.execute(
            "INSERT INTO PREDICTIONS (PREDICTION_ID, PHARMACIE_ID, CLASSE_PREDITE, PROBABILITES, CONFIANCE) "
            "SELECT %s, %s, %s, PARSE_JSON(%s), %s",
            (str(uuid.uuid4()), pharmacie_id, classe_predite, json.dumps(probabilites_dict), float(confiance))
        )
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        st.warning(f"Snowflake insert échoué : {e}")
        return False

# ── Spectrogramme Mel ───────────────────────────────────────────────────────
def plot_mel_spectrogram(audio):
    fig, ax = plt.subplots(figsize=(10, 3))
    S    = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=1024, hop_length=512, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img  = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=SR, ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Spectrogramme Mel')
    plt.tight_layout()
    return fig

# ── Page principale ─────────────────────────────────────────────────────────
def diagnostic_page():
    model, augmenter, device = load_model()

    # Sidebar
    with st.sidebar:
        st.subheader("Cabine Tessan")
        pharmacie_id = st.selectbox("Pharmacie", PHARMACIES)

        st.markdown("---")
        st.subheader("🔌 Snowflake")

        if "_sf" not in st.query_params:
            totp = st.text_input("Code MFA (6 chiffres)", max_chars=6, type="password")
            if st.button("Connecter") and totp:
                try:
                    get_snowflake_connection(totp)     # met en cache partagé
                    st.query_params["_sf"] = totp      # persiste dans l'URL
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Connexion échouée : {e}")
        else:
            st.success("✅ Connecté")
            if st.button("Déconnecter"):
                del st.query_params["_sf"]
                st.rerun()

    # Main
    st.title("Détection des Maladies Respiratoires")
    st.markdown("**Analyse assistée par IA des sons respiratoires en cabine médicale**")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("1. Acquisition du son")
        st.info("Simulez l'action du stéthoscope connecté de la cabine Tessan.")
        uploaded_file = st.file_uploader("Chargez un enregistrement d'auscultation (.wav)", type=['wav'])

        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            with st.spinner('Extraction des caractéristiques audio...'):
                audio, _ = librosa.load(uploaded_file, sr=SR)
                audio_preprocessed = preprocess_audio(audio)
                st.subheader("Signature Acoustique")
                st.pyplot(plot_mel_spectrogram(audio_preprocessed))

    with col2:
        st.header("2. Pré-diagnostic IA")
        if uploaded_file is not None:
            if model is None:
                st.error("Modèle non chargé.")
            else:
                with st.spinner('Analyse par le modèle en cours...'):
                    predictions    = predict(audio_preprocessed, model, augmenter, device)
                    classe_predite = max(predictions, key=predictions.get)
                    confiance      = predictions[classe_predite]

                st.metric(
                    label="Diagnostic principal estimé",
                    value=CLASS_MAP[classe_predite],
                    delta=f"{confiance:.1%} de confiance"
                )

                df_preds = pd.DataFrame({
                    'Pathologie':    [CLASS_MAP[c] for c in CLASSES],
                    'Probabilité (%)': [predictions[c] * 100 for c in CLASSES]
                })
                fig_bar = px.bar(
                    df_preds, x='Probabilité (%)', y='Pathologie',
                    orientation='h', color='Pathologie',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_bar.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_bar, width='stretch')

                st.subheader("Recommandation Clinique")
                icon, level, msg = CLINICAL_RECS[classe_predite]
                getattr(st, level)(f"{icon} {msg}")

                # ── Comparaison avec sons de référence ───────────────────
                refs = load_reference_embeddings()
                if refs is not None:
                    embedding = get_embedding(audio_preprocessed, model, augmenter, device)
                    sims      = compare_to_references(embedding, refs)
                    st.subheader("Comparaison avec sons de référence")
                    df_sim = pd.DataFrame({
                        'Pathologie':   [CLASS_MAP[c] for c in CLASSES],
                        'Similarité (%)': [sims.get(c, 0) * 100 for c in CLASSES],
                    })
                    fig_sim = px.bar(
                        df_sim, x='Similarité (%)', y='Pathologie',
                        orientation='h',
                        color='Pathologie',
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig_sim.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_sim, width='stretch')
                    closest = max(sims, key=sims.get)
                    st.caption(f"Son le plus proche du dataset de référence : **{CLASS_MAP[closest]}** ({sims[closest]:.0%} similarité normalisée)")

                saved = insert_prediction(pharmacie_id, classe_predite, predictions, confiance)
                if saved:
                    st.caption("✅ Prédiction enregistrée dans Snowflake")
                elif 'sf_conn' not in st.session_state:
                    st.caption("⚠️ Non connecté à Snowflake — prédiction non sauvegardée")
        else:
            st.write("En attente de l'enregistrement du patient...")

    st.markdown("---")
    st.caption("Hackathon Tessan x Snowflake | Pipeline de données et de déploiement sécurisé")