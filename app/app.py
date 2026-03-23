import base64
import streamlit as st
from PIL import Image

from views.diagnostic_page import diagnostic_page
from views.dashboard_page import dashboard_page


favicon_path = "app/assets/favicon.jpg"

try : 
    favicon = Image.open(favicon_path)
except FileNotFoundError:
    favicon = "🩺"

st.set_page_config(
    page_title="Tessan - Diagnostic Respiratoire",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        return "" 

logo_tessan_b64 = get_base64_image("app/assets/tessan.svg")
icon_diag_b64 = get_base64_image("app/assets/diagnostic.svg")
icon_dash_b64 = get_base64_image("app/assets/dashboard.svg")

page = st.query_params.get("page", "diagnostic").lower()
sf_param = st.query_params.get("_sf", "")
sf_qs = f"&_sf={sf_param}" if sf_param else ""

try:
    with open("app/css/styles.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Fichier CSS introuvable. Vérifiez le chemin 'app/css/styles.css'.")

st.markdown(f"""
<div class="floating-island">
<img src="data:image/svg+xml;base64,{logo_tessan_b64}" class="logo" alt="Logo Tessan">
<a href="/?page=diagnostic{sf_qs}" target="_self" class="nav-link"><img src="data:image/svg+xml;base64,{icon_diag_b64}" alt="Diagnostic"> Diagnostic</a>
<a href="/?page=dashboard{sf_qs}" target="_self" class="nav-link"><img src="data:image/svg+xml;base64,{icon_dash_b64}" alt="Dashboard"> Dashboard</a>
</div>
""", unsafe_allow_html=True)

if page == "diagnostic":
    diagnostic_page()
elif page == "dashboard":
    dashboard_page()
else:
    st.error("Page non trouvée. Veuillez vérifier l'URL.")