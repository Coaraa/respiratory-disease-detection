import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn  
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import streamlit as st
from pathlib import Path
from scipy.signal import butter, filtfilt
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'app' / 'views'))

import grad_cam

# GPUAugmenter importé depuis notebooks/ConvNeXt.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'notebooks'))
from ConvNeXt import GPUAugmenter, RespiratoryModel
sys.path.insert(0, str(Path(__file__).parent.parent))
from snowflake_conn import get_snowflake_connection

load_dotenv(Path(__file__).parent.parent.parent / '.env')

# Constantes audio
SR         = 22050
TARGET_LEN = SR * 6  # 132 300 samples
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

# Preprocessing
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
    return pad_or_crop(audio)

# Chargement modèle 
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

# Inférence 
def predict(audio_array, model, augmenter, device):
    tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        spec    = augmenter(tensor, augment=False)
        outputs = model(spec)
        probs   = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
    return dict(zip(CLASSES, probs.tolist()))

# Chargement des embeddings 
def get_embedding(audio_array, model, augmenter, device):
    tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        spec     = augmenter(tensor, augment=False)
        features = model.cnn(spec)                   
        return features.squeeze().cpu().numpy()      

# Embeddings de référence
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
    sims  = {cls: cosine_similarity(embedding, refs[cls]) for cls in CLASSES if cls in refs}
    lo, hi = min(sims.values()), max(sims.values())
    span  = hi - lo if hi > lo else 1.0
    return {cls: (v - lo) / span for cls, v in sims.items()}

def get_sf_connection():
    """Récupère la connexion partagée via le TOTP stocké dans l'URL."""
    totp = st.query_params.get("_sf")
    if totp:
        return get_snowflake_connection(totp)
    return None

# Insert Snowflake
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

# Spectrogramme Mel 
def plot_mel_spectrogram(audio):
    fig, ax = plt.subplots(figsize=(10, 3))
    S    = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=1024, hop_length=512, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img  = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=SR, ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Spectrogramme Mel')
    plt.tight_layout()
    return fig

# Page principale
def diagnostic_page():
    model, augmenter, device = load_model()

    # Sidebar
    with st.sidebar:
        pharmacie_id = st.selectbox("Pharmacie", PHARMACIES)

        st.markdown("---")
        st.subheader("🔌 Snowflake")

        if "_sf" not in st.query_params:
            totp = st.text_input("Code MFA (6 chiffres)", max_chars=6, type="password")
            if st.button("Connecter") and totp:
                try:
                    get_snowflake_connection(totp)     
                    st.query_params["_sf"] = totp     
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Connexion échouée : {e}")
        else:
            st.success("✅ Connecté")
            if st.button("Déconnecter"):
                del st.query_params["_sf"]
                st.rerun()

   
    st.title("Diagnostic Respiratoire")

    st.markdown("---")

    # Zone d'upload 
    uploaded_file = st.file_uploader(
        "Déposez un enregistrement d'auscultation (.wav)",
        type=['wav'],
        label_visibility='collapsed',
    )

    if uploaded_file is None:
        st.info("En attente d'un enregistrement WAV.")
        return

    # Chargement et preprocessing 
    with st.spinner('Prétraitement audio…'):
        audio, _ = librosa.load(uploaded_file, sr=SR)
        audio_preprocessed = preprocess_audio(audio)

    st.audio(uploaded_file, format='audio/wav')
    st.markdown("---")

    if model is None:
        st.error("Modèle non chargé.")
        return

    # Inférence et Grad-CAM 
    with st.spinner('Analyse IA en cours…'):
        predictions    = predict(audio_preprocessed, model, augmenter, device)
        classe_predite = max(predictions, key=predictions.get)
        confiance      = predictions[classe_predite]

        S       = librosa.feature.melspectrogram(y=audio_preprocessed, sr=SR, n_fft=1024, hop_length=512, n_mels=128)
        S_dB    = librosa.power_to_db(S, ref=np.max)
        img_cv2 = (255 * (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())).astype(np.uint8)
        img_cv2 = np.flipud(img_cv2)
        img_tensor  = grad_cam.preprocess_image(img_cv2).to(device)
        model.eval()
        with torch.no_grad():
            preds = model(img_tensor)
        class_index = preds.argmax(dim=1).item()
        heatmap     = grad_cam.compute_gradcam(model, img_tensor, class_index, conv_layer_name="cnn.features.7")
        output_img  = grad_cam.overlay_heatmap(img_cv2, heatmap)

    # Mise en page deux colonnes 
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Diagnostic principal
        DISEASE_COLORS = {
            'asthma': '#F59E0B', 'copd': '#EF4444', 'bronchial': '#8B5CF6',
            'pneumonia': '#F97316', 'healthy': '#10B981',
        }
        color = DISEASE_COLORS.get(classe_predite, '#6B7280')
        st.markdown(f"""
        <div style="background:{color}18;border:1.5px solid {color}40;
                    border-radius:12px;padding:20px 24px;margin-bottom:16px">
            <div style="font-size:13px;color:#6B7280;font-weight:600;
                        text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">
                Diagnostic principal
            </div>
            <div style="font-size:32px;font-weight:800;color:{color}">
                {CLASS_MAP[classe_predite]}
            </div>
            <div style="font-size:15px;color:#374151;margin-top:4px">
                Confiance : <b>{confiance:.1%}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommandation clinique
        icon, level, msg = CLINICAL_RECS[classe_predite]
        getattr(st, level)(f"{icon} {msg}")

        st.markdown("#### Probabilités par pathologie")
        df_preds = pd.DataFrame({
            'Pathologie':      [CLASS_MAP[c] for c in CLASSES],
            'Probabilité (%)': [predictions[c] * 100 for c in CLASSES],
        })
        fig_bar = px.bar(
            df_preds, x='Probabilité (%)', y='Pathologie',
            orientation='h', color='Pathologie',
            color_discrete_map={CLASS_MAP[k]: v for k, v in DISEASE_COLORS.items()},
        )
        fig_bar.update_layout(
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=220,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

        # Similarité de référence
        refs = load_reference_embeddings()
        if refs is not None:
            embedding = get_embedding(audio_preprocessed, model, augmenter, device)
            sims      = compare_to_references(embedding, refs)
            st.markdown("#### Similarité avec le dataset de référence")
            df_sim = pd.DataFrame({
                'Pathologie':     [CLASS_MAP[c] for c in CLASSES],
                'Similarité (%)': [sims.get(c, 0) * 100 for c in CLASSES],
            })
            fig_sim = px.bar(
                df_sim, x='Similarité (%)', y='Pathologie',
                orientation='h', color='Pathologie',
                color_discrete_map={CLASS_MAP[k]: v for k, v in DISEASE_COLORS.items()},
            )
            fig_sim.update_layout(
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=220,
            )
            st.plotly_chart(fig_sim, use_container_width=True, config={'displayModeBar': False})
        insert_prediction(pharmacie_id, classe_predite, predictions, confiance)

    with col_right:
        st.markdown("#### Spectrogramme Mel")
        fig_mel, ax = plt.subplots(figsize=(6, 2.8))
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=SR, ax=ax, cmap='magma')
        fig_mel.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('')
        plt.tight_layout(pad=0.5)
        st.pyplot(fig_mel, use_container_width=True)

        st.markdown("#### Grad-CAM")
        st.image(output_img, use_container_width=True,
                 caption="Régions fréquentielles et temporelles activant le diagnostic")