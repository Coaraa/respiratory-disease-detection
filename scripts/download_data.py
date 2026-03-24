import kagglehub
import shutil
import os

# Téléchargement du dataset
path = kagglehub.dataset_download("mohammedtawfikmusaed/asthma-detection-dataset-version-2")
print("Dataset téléchargé :", path)

inner = os.path.join(path, "Asthma Detection Dataset Version 2", "Asthma Detection Dataset Version 2")

# Dossier data/ à la racine du projet
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")

os.makedirs(data_dir, exist_ok=True)

# Copie des 5 dossiers de classes dans data/
for class_folder in os.listdir(inner):
    src = os.path.join(inner, class_folder)
    dst = os.path.join(data_dir, class_folder)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  Copié : {class_folder} → data/{class_folder}")

print(f"\nDone. Dossier data/ prêt dans : {data_dir}")
