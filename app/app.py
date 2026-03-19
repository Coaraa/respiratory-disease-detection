import os
import streamlit as st

from views.diagnostic_page import diagnostic_page
from views.dashboard_page import dashboard_page

import streamlit as st
import os

# 1. Configuration de la page (DOIT toujours être le premier appel Streamlit)
st.set_page_config(
    page_title="Tessan - Diagnostic Respiratoire",
    page_icon="🩺",
    layout="wide"
)

# 2. Gestion des chemins des logos
DOSSIER_COURANT = os.path.dirname(os.path.abspath(__file__))
chemin_logo_tessan = os.path.join(DOSSIER_COURANT, "assets", "tessan.png")
url_logo_snowflake = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Snowflake_Logo.svg/1200px-Snowflake_Logo.svg.png"

# --- 3. En-tête : Logos ---
# On crée 3 colonnes : les logos aux extrémités et un grand espace au milieu
col_logo_gauche, col_titre, col_logo_droite = st.columns([1, 6, 1])

with col_logo_gauche:
    try:
        st.image(chemin_logo_tessan, use_container_width=True)
    except Exception:
        st.error("Logo Tessan introuvable")

with col_titre:
    # On peut mettre un titre centré au milieu si on le souhaite
    st.markdown("<h2 style='text-align: center; color: #2E86C1;'>TESSAN x Snowflake Hackathon</h2>", unsafe_allow_html=True)

with col_logo_droite:
    st.image(url_logo_snowflake, use_container_width=True)


# --- 4. Barre de navigation horizontale ---
st.markdown("<br>", unsafe_allow_html=True) # Ajoute un petit saut de ligne pour aérer

page_selectionnee = st.radio(
    "Menu de navigation :",
    ["🩺 Diagnostic IA", "📊 Dashboard Épidémiologique"],
    horizontal=True,
    label_visibility="collapsed" # Cache le titre "Menu de navigation :" pour faire plus propre
)

st.markdown("---") # Ligne de séparation sous le menu

# --- 5. Routage vers le contenu des pages ---
if page_selectionnee == "🩺 Diagnostic IA":
    # Assure-toi d'avoir une fonction render_diagnostic_page() dans ton fichier diagnostic_page.py
    diagnostic_page()
    
elif page_selectionnee == "📊 Dashboard Épidémiologique":
    dashboard_page()