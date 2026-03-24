import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torchvision.models import convnext_small
from .grad_cam import GradCAM, overlay_grad_cam
import cv2
import tempfile
import os

class GPUAugmenter(nn.Module):
    def __init__(self, sr=22050):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=512, n_mels=128)
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80.0)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)
        self.pitch_shift = T.PitchShift(sample_rate=sr, n_steps=2)

    def forward(self, x, augment=True):
        x = torch.nan_to_num(x)
        if augment:
            if torch.rand(1) < 0.3: x = x + 0.001 * torch.randn_like(x)
            if torch.rand(1) < 0.2: x = self.pitch_shift(x)
            
        spec = self.amplitude_to_db(self.mel_spec(x) + 1e-10)
        
        if augment:
            if torch.rand(1) < 0.4: spec = self.freq_mask(spec)
            if torch.rand(1) < 0.4: spec = self.time_mask(spec)

        # Standardisation de l'image pour ConvNeXt
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        return spec

class RespiratoryFusionModel(nn.Module):
    def __init__(self, num_classes=5, num_features=16):
        super().__init__()
        self.cnn = convnext_small(weights="DEFAULT")
        
        # Adaptation 1 canal
        old_conv = self.cnn.features[0][0]
        self.cnn.features[0][0] = nn.Conv2d(1, old_conv.out_channels, kernel_size=4, stride=4)
        with torch.no_grad():
            self.cnn.features[0][0].weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        
        self.cnn.classifier = nn.Sequential(
            nn.LayerNorm((768, 1, 1), eps=1e-6),
            nn.Flatten(1)
        )

        # Branche MLP pour les features acoustiques
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Classifieur Final (Fusion du CNN et du MLP)
        self.final_classifier = nn.Sequential(
            nn.Linear(832, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, spec, feat):
        cnn_out = self.cnn(spec)
        mlp_out = self.mlp(feat)
        
        combined = torch.cat((cnn_out, mlp_out), dim=1)
        return self.final_classifier(combined)

def diagnostic_page():         
    # En-tête de l'application
    st.title("Détection des Maladies Respiratoires")
    st.markdown("**Analyse assistée par IA des sons respiratoires en cabine médicale**")
    st.markdown("---")

    @st.cache_resource
    def load_model():
        model = RespiratoryFusionModel(num_classes=5, num_features=16)
        model.load_state_dict(torch.load("models/best_model_convnext_with_features.pth", map_location=torch.device('cpu')))
        model.eval()
        return model

    model = load_model()
    classes = ['Asthme', 'BPCO', 'Bronchique', 'Pneumonie', 'Sain']

    def plot_mel_spectrogram(audio_data, sr):
        """Génère et affiche le spectrogramme Mel du fichier audio."""
        fig, ax = plt.subplots(figsize=(10, 4))
        # Paramètres de base recommandés par le sujet
        S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Spectrogramme Mel')
        return fig, S_dB

    # Colonnes pour structurer l'interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("1. Acquisition du son")
        st.info("Simulez l'action du stéthoscope connecté de la cabine Tessan.")
        
        # Upload du fichier WAV
        uploaded_file = st.file_uploader("Chargez un enregistrement d'auscultation (.wav)", type=['wav'])
        
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        with col1:
            st.audio(tmp_path, format='audio/wav')
            
            with st.spinner('Extraction des caractéristiques audio...'):
                # Chargement de l'audio avec librosa
                y, sr = librosa.load(tmp_path, sr=22050)
                
                st.subheader("Signature Acoustique")
                # Affichage du spectrogramme Mel
                fig_mel, spec_db = plot_mel_spectrogram(y, sr)
                st.pyplot(fig_mel)

        with col2:
            st.header("2. Pré-diagnostic IA")
            with st.spinner('Analyse par le modèle en cours...'):
                
                # Preprocessing
                y, sr = librosa.load(tmp_path, sr=22050)
                spec = T.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=512, n_mels=128)(torch.tensor(y).unsqueeze(0))
                spec = T.AmplitudeToDB(stype='power', top_db=80.0)(spec)
                spec = (spec - spec.mean()) / (spec.std() + 1e-8)
                spec = spec.unsqueeze(0) # Add batch dimension

                # Fake features for now
                feats = torch.randn(1, 16)

                # Prediction
                output = model(spec, feats)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                
                # Grad-CAM
                target_layer = model.cnn.features[7]
                grad_cam = GradCAM(model, target_layer)
                
                cam = grad_cam(spec, feats)

                # Convert spectrogram to image for overlay
                spec_img = librosa.display.specshow(spec.squeeze().detach().numpy(), sr=sr, x_axis='time', y_axis='mel', fmax=8000)
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig("spec.png", bbox_inches='tight', pad_inches=0)
                plt.close()

                spec_img = cv2.imread("spec.png")
                spec_img = np.float32(spec_img) / 255
                
                overlaid_image = overlay_grad_cam(spec_img, cam)
                
                st.subheader("Visualisation de l'attention du modèle (Grad-CAM)")
                st.image(overlaid_image, caption="Le heatmap de couleur vive indique où le modèle se concentre pour prendre sa décision.", use_column_width=S)


                # Formater les données pour Plotly
                df_preds = pd.DataFrame({
                    'Pathologie': classes,
                    'Probabilité (%)': [p * 100 for p in probabilities.detach().tolist()]
                })
                
                # Trouver la classe prédite
                classe_predite = df_preds.loc[df_preds['Probabilité (%)'].idxmax()]['Pathologie']
                confiance = df_preds['Probabilité (%)'].max()
                
                st.metric(label="Diagnostic principal estimé", value=classe_predite, delta=f"{confiance:.1f}% de confiance")
                
                # Affichage du graphique en barres
                st.subheader("Répartition des probabilités")
                fig_bar = px.bar(
                    df_preds, 
                    x='Probabilité (%)', 
                    y='Pathologie', 
                    orientation='h',
                    color='Pathologie',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_bar.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Recommandation clinique
                st.subheader("Recommandation Clinique")
                if classe_predite == "Asthme":
                    st.warning("⚠️ Suspicion d'asthme (sifflements détectés). Recommandation : Consultation médicale pour confirmation et prescription éventuelle de bronchodilatateurs.")
                elif classe_predite == "Sain":
                    st.success("✅ Murmure vésiculaire régulier. Aucune anomalie respiratoire majeure détectée.")
                else:
                    st.error(f"🚨 Suspicion de {classe_predite}. Recommandation : Examen approfondi requis.")
            
            # Clean up the temporary file
            os.remove(tmp_path)
                    
    else:
        with col2:
            st.write("En attente de l'enregistrement du patient...")

    st.markdown("---")
    st.caption("Hackathon Tessan x Snowflake | Pipeline de données et de déploiement sécurisé")
