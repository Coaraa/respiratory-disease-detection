import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px

def diagnostic_page():         
    # En-tête de l'application
    st.title("Détection des Maladies Respiratoires")
    st.markdown("**Analyse assistée par IA des sons respiratoires en cabine médicale**")
    st.markdown("---")

    def plot_mel_spectrogram(audio_data, sr):
        """Génère et affiche le spectrogramme Mel du fichier audio."""
        fig, ax = plt.subplots(figsize=(10, 4))
        # Paramètres de base recommandés par le sujet
        S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title('Spectrogramme Mel')
        return fig

    def mock_snowpark_prediction():
        """Simule la réponse de l'UDF Snowflake (à remplacer par ton vrai modèle)."""
        # Classes ciblées par le hackathon
        classes = ['Asthme', 'BPCO', 'Bronchique', 'Pneumonie', 'Sain']
        # Probabilités fictives pour la démo
        probs = [0.62, 0.18, 0.07, 0.10, 0.03]
        return dict(zip(classes, probs))

    # Colonnes pour structurer l'interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("1. Acquisition du son")
        st.info("Simulez l'action du stéthoscope connecté de la cabine Tessan.")
        
        # Upload du fichier WAV
        uploaded_file = st.file_uploader("Chargez un enregistrement d'auscultation (.wav)", type=['wav'])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            with st.spinner('Extraction des caractéristiques audio...'):
                # Chargement de l'audio avec librosa
                y, sr = librosa.load(uploaded_file, sr=22050)
                
                st.subheader("Signature Acoustique")
                # Affichage du spectrogramme Mel
                fig_mel = plot_mel_spectrogram(y, sr)
                st.pyplot(fig_mel)

    with col2:
        st.header("2. Pré-diagnostic IA")
        if uploaded_file is not None:
            with st.spinner('Analyse par le modèle Snowflake en cours...'):
                # Appel fictif au modèle
                predictions = mock_snowpark_prediction()
                
                # Formater les données pour Plotly
                df_preds = pd.DataFrame({
                    'Pathologie': list(predictions.keys()),
                    'Probabilité (%)': [p * 100 for p in predictions.values()]
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
                st.plotly_chart(fig_bar, width='stretch')
                
                # Recommandation clinique
                st.subheader("Recommandation Clinique")
                if classe_predite == "Asthme":
                    st.warning("⚠️ Suspicion d'asthme (sifflements détectés). Recommandation : Consultation médicale pour confirmation et prescription éventuelle de bronchodilatateurs.")
                elif classe_predite == "Sain":
                    st.success("✅ Murmure vésiculaire régulier. Aucune anomalie respiratoire majeure détectée.")
                else:
                    st.error(f"🚨 Suspicion de {classe_predite}. Recommandation : Examen approfondi requis.")
                    
        else:
            st.write("En attente de l'enregistrement du patient...")

    st.markdown("---")
    st.caption("Hackathon Tessan x Snowflake | Pipeline de données et de déploiement sécurisé")
