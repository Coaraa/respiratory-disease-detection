import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

@st.cache_data
def load_mock_predictions():
    """
    Génère de fausses données simulant la table PREDICTIONS de Snowflake.
    Structure attendue : timestamp, pharmacie_id, classe_predite, probabilites, confiance.
    """
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=i) for i in range(30)]
    pharmacies = [f"PHARM-{str(i).zfill(3)}" for i in range(1, 21)]
    classes = ['Asthme', 'BPCO', 'Bronchique', 'Pneumonie', 'Sain']
    
    data = []
    for _ in range(500): # Génération de 500 fausses prédictions
        data.append({
            'timestamp': np.random.choice(dates),
            'pharmacie_id': np.random.choice(pharmacies),
            'region': np.random.choice(['Île-de-France', 'PACA', 'Nouvelle-Aquitaine', 'Auvergne-Rhône-Alpes', 'Bretagne']),
            'classe_predite': np.random.choice(classes, p=[0.25, 0.35, 0.10, 0.15, 0.15]),
            'confiance': np.random.uniform(0.65, 0.99)
        })
    return pd.DataFrame(data)

def dashboard_page():
    """
    Rendu de la page du Dashboard Épidémiologique.
    """
    st.title("📊 Tableau de Bord Épidémiologique")
    st.markdown("Surveillance en temps réel des détections respiratoires sur le réseau de cabines Tessan.")
    st.markdown("---")

    # Chargement des données (simulées pour l'instant)
    df = load_mock_predictions()

    # --- 1. KPIs Principaux ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total des analyses (30j)", f"{len(df)}")
    col2.metric("Cas d'Asthme", f"{len(df[df['classe_predite'] == 'Asthme'])}")
    col3.metric("Urgences Pneumonie", f"{len(df[df['classe_predite'] == 'Pneumonie'])}", delta="+12%", delta_color="inverse")
    col4.metric("Confiance moyenne IA", f"{df['confiance'].mean():.1%}")

    st.markdown("---")

    # --- 2. Graphiques d'analyse ---
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Évolution temporelle")
        st.caption("Tendances sur les 30 derniers jours")
        # Grouper par date et par classe
        df_trend = df.groupby([df['timestamp'].dt.date, 'classe_predite']).size().reset_index(name='count')
        fig_trend = px.line(df_trend, x='timestamp', y='count', color='classe_predite', markers=True)
        st.plotly_chart(fig_trend, width='stretch')

    with col_chart2:
        st.subheader("Répartition Géographique")
        st.caption("Détection des pics épidémiologiques par région")
        # Grouper par région et par classe
        df_region = df.groupby(['region', 'classe_predite']).size().reset_index(name='count')
        fig_region = px.bar(df_region, x='region', y='count', color='classe_predite', barmode='stack')
        st.plotly_chart(fig_region, width='stretch')

    # --- 3. Vue détaillée (Tableau) ---
    st.subheader("Dernières prédictions (Vue détaillée)")
    st.caption("Aperçu de la table PREDICTIONS stockée dans Snowflake")
    # Affichage des 10 dernières lignes triées par date
    st.dataframe(df.sort_values(by='timestamp', ascending=False).head(10), width='stretch')