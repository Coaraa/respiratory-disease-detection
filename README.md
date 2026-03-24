# Détection des Maladies Respiratoires par Analyse de Sons Respiratoires 

***TESSAN x Snowflake Hackathon Gen IA*** 


## 📖 Description du projet

Notre projet est un système d'intelligence artificielle conçu pour détecter des maladies respiratoires à partir de sons de respiration (auscultation pulmonaire). Cette solution a pour objectif d'être intégrée directement dans les cabines médicales connectées Tessan, en utilisant l'écosystème Snowflake comme plateforme de données.

## Fonctionnalités

* **Pré-diagnostic en temps réel** : Analyse des biomarqueurs respiratoires et calcul des probabilités pour 5 classes : Asthme, BPCO, Bronchite, Pneumonie et Patient Sain.
* **Interface interactive** : Application web permettant aux médecins d'uploader un son respiratoire (`.wav`) et d'obtenir une recommandation clinique instantanée.
* **Explicabilité médicale** : Mise en évidence des zones du spectrogramme (via Grad-CAM ou SHAP) qui ont été déterminantes pour justifier la prédiction du modèle.
* **Surveillance épidémiologique** : Sauvegarde des prédictions dans Snowflake pour créer un tableau de bord capable de détecter des pics de maladies par région.

## 🚀 Lancement du projet

## 👥 Auteurs

- LEFEVRE Victor
- ROZIER Loïc
- SITHIDEJ Clara