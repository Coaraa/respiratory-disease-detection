SR         = 22050
TARGET_LEN = SR * 6   # 132 300 samples — 6 secondes

CLASSES = ['asthma', 'bronchial', 'copd', 'healthy', 'pneumonia']

CLASS_MAP = {
    'asthma':    'Asthme',
    'bronchial': 'Bronchite',
    'copd':      'BPCO',
    'healthy':   'Sain',
    'pneumonia': 'Pneumonie',
}

CLASSES_FR = ['Asthme', 'BPCO', 'Bronchite', 'Pneumonie', 'Sain']

# Clés françaises → couleur hex (dashboard)
DISEASE_COLORS = {
    'Asthme':    '#F59E0B',
    'BPCO':      '#EF4444',
    'Bronchite': '#8B5CF6',
    'Pneumonie': '#F97316',
    'Sain':      '#10B981',
}

# Clés anglaises → même palette (diagnostic)
DISEASE_COLORS_EN = {en: DISEASE_COLORS[fr] for en, fr in CLASS_MAP.items()}

FOLIUM_ICON_COLOR = {
    'Asthme':    'orange',
    'BPCO':      'red',
    'Bronchite': 'purple',
    'Pneumonie': 'darkred',
    'Sain':      'green',
}

CLINICAL_RECS = {
    'asthma':    ("⚠️", "warning", "Suspicion d'asthme (sifflements détectés). Consultation médicale recommandée pour confirmation et prescription éventuelle de bronchodilatateurs."),
    'bronchial': ("⚠️", "warning", "Suspicion de syndrome bronchite. Bilan ORL/pulmonaire conseillé."),
    'copd':      ("🚨", "error",   "Suspicion de BPCO. Examen spirométrique urgent recommandé."),
    'healthy':   ("✅", "success", "Murmure vésiculaire régulier. Aucune anomalie respiratoire majeure détectée."),
    'pneumonia': ("🚨", "error",   "Suspicion de pneumonie. Consultation médicale urgente et radiographie pulmonaire recommandées."),
}

PHARMACIES = [
    "PHARM_BORDEAUX_001", "PHARM_PARIS_002", "PHARM_LYON_003",
    "PHARM_MARSEILLE_004", "PHARM_TOULOUSE_005",
]
