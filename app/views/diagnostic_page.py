import sys
import json
import cv2
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'app' / 'views'))
import grad_cam

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'notebooks'))
from ConvNeXt import GPUAugmenter, RespiratoryModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SR, TARGET_LEN, CLASSES, CLASS_MAP, PHARMACIES, CLINICAL_RECS, DISEASE_COLORS_EN
from audio_utils import preprocess_audio, get_embedding, compare_to_references
from snowflake_conn import get_snowflake_connection, render_snowflake_sidebar

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

@st.cache_resource
def load_reference_embeddings():
    npz_path = Path(__file__).parent.parent.parent / 'models' / 'reference_embeddings.npz'
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    return {cls: data[cls] for cls in CLASSES if cls in data}

def get_sf_connection():
    totp = st.query_params.get("_sf")
    return get_snowflake_connection(totp) if totp else None

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
        render_snowflake_sidebar()

   
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
        color = DISEASE_COLORS_EN.get(classe_predite, '#6B7280')
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
            color_discrete_map={CLASS_MAP[k]: v for k, v in DISEASE_COLORS_EN.items()},
        )
        fig_bar.update_layout(
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=220,
        )
        st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False})

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
                color_discrete_map={CLASS_MAP[k]: v for k, v in DISEASE_COLORS_EN.items()},
            )
            fig_sim.update_layout(
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=220,
            )
            st.plotly_chart(fig_sim, width='stretch', config={'displayModeBar': False})
        insert_prediction(pharmacie_id, classe_predite, predictions, confiance)

    with col_right:
        st.markdown("#### Spectrogramme Mel")
        fig_mel, ax = plt.subplots(figsize=(6, 2.8))
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=SR, ax=ax, cmap='magma')
        fig_mel.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('')
        plt.tight_layout(pad=0.5)
        st.pyplot(fig_mel, width='stretch')

        st.markdown("#### Grad-CAM")
        st.image(output_img, width='stretch',
                 caption="Régions fréquentielles et temporelles activant le diagnostic")