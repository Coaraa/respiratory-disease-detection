import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px

from views.diagnostic_page import diagnostic_page

# Configuration de la page
st.set_page_config(
    page_title="Tessan - Diagnostic Respiratoire IA",
    page_icon="🩺",
    layout="wide"
)

with open("css/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


if "page_actuelle" not in st.session_state:
    st.session_state.page_actuelle = "diagnostic"


with st.sidebar:
        #st.image("assets/logo.png", width=120)

        st.title("TESSAN")
        st.divider()

        if st.button("🩺 Diagnostic", width='stretch', type="primary" if st.session_state.page_actuelle == "diagnostic" else "secondary"):
            st.session_state.page_actuelle = "diagnostic"
            st.rerun()

        if st.button("🗺️ Tableau de bord", width='stretch', type="primary" if st.session_state.page_actuelle == "dashboard" else "secondary"):
            st.session_state.page_actuelle = "dashboard"
            st.rerun()
            
 



if st.session_state.page_actuelle == "diagnostic":
        diagnostic_page()
else : 
    st.write("Page non trouvée. Veuillez sélectionner une page valide.")