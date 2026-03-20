from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import ConvNeXt_with_features
import ConvNeXt
import CNN

def preparation_donnees(directory):
    try:
        X_features = np.load(directory + "X_features.npy")
        X = np.load(directory + "X.npy")
        y = np.load(directory + "y.npy")
        print(f"Données chargées : X={X.shape}, y={y.shape}")
    except:
        print("Erreur : Fichier non trouvé. Vérifie le nom de tes signaux bruts.")

    # Encodage des labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split en train, validation et test
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(split1.split(X, y_encoded))

    X_train, X_temp = X[train_idx], X[temp_idx]
    X_features_train, X_features_temp = X_features[train_idx], X_features[temp_idx]
    y_train, y_temp = y_encoded[train_idx], y_encoded[temp_idx]

    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(split2.split(X_temp, y_temp))

    X_val, X_test = X_temp[val_idx], X_temp[test_idx]
    X_features_val, X_features_test = X_features_temp[val_idx], X_features_temp[test_idx]
    y_val, y_test = y_temp[val_idx], y_temp[test_idx]
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    scaler = StandardScaler()

    # On apprend les paramètres uniquement sur le TRAIN
    X_features_train = scaler.fit_transform(X_features_train)

    # On applique les MÊMES paramètres (moyenne/std du train) au reste
    X_features_val = scaler.transform(X_features_val)
    X_features_test = scaler.transform(X_features_test)

    return X_train, X_features_train, y_train, X_val, X_features_val, y_val, X_test, X_features_test, y_test, le



def evaluate_model(y_test, y_test_pred, y_test_proba, le, afficher_resultats=True):
    if not afficher_resultats:
        accuracy = accuracy_score(y_test, y_test_pred)
        macro_f1 = f1_score(y_test, y_test_pred, average='macro')

        y_test_onehot = np.eye(len(le.classes_))[y_test]

        auc_scores = []
        for i, class_name in enumerate(le.classes_):
            try:
                auc_score = roc_auc_score(y_test_onehot[:, i], y_test_proba[:, i])
                auc_scores.append(auc_score)
                print(f"{class_name:15s}: AUC-ROC = {auc_score:.4f}")
            except Exception:
                print(f"{class_name:15s}: AUC-ROC = N/A (classe insuffisante)")

        mean_auc = np.mean(auc_scores) if auc_scores else 0

        return accuracy, macro_f1, mean_auc

    print("="*70)
    print("ÉVALUATION DU MODÈLE SUR LE JEU DE TEST")
    print("="*70)

    accuracy = accuracy_score(y_test, y_test_pred)
    macro_f1 = f1_score(y_test, y_test_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_test_pred, average='weighted')

    print(f"\nAccuracy globale:        {accuracy:.4f}")
    print(f"Macro F1-score:          {macro_f1:.4f}  <- Métrique clé")
    print(f"Weighted F1-score:       {weighted_f1:.4f}")

    print("\n" + "="*70)
    print("RAPPORT DE CLASSIFICATION PAR CLASSE")
    print("="*70)
    print("\n(Sensibilité = Recall = % de vrais positifs détectés par classe)")
    print(classification_report(y_test, y_test_pred,
                            target_names=le.classes_,
                            digits=4,
                            zero_division=0))

    cm = confusion_matrix(y_test, y_test_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                cbar_kws={'label': 'Nombre de samples'}, ax=ax)
    ax.set_xlabel('Prédiction')
    ax.set_ylabel('Vérité')
    ax.set_title('Matrice de Confusion - Test Set\n(Les faux négatifs sont critiques en médecine)')
    plt.tight_layout()
    plt.show()

    print("\n" + "="*70)
    print("ANALYSE DES ERREURS - FAUX NÉGATIFS PAR CLASSE")
    print("="*70)
    for i, class_name in enumerate(le.classes_):
        fn = cm[i, :].sum() - cm[i, i]
        tp = cm[i, i]
        sensibilite = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"{class_name:15s}: FN={fn:3d}, TP={tp:3d}, Sensibilité={sensibilite:.4f}")

    print("\n" + "="*70)
    print("AUC-ROC PAR CLASSE")
    print("="*70)

    y_test_onehot = np.eye(len(le.classes_))[y_test]

    auc_scores = []
    for i, class_name in enumerate(le.classes_):
        try:
            auc_score = roc_auc_score(y_test_onehot[:, i], y_test_proba[:, i])
            auc_scores.append(auc_score)
            print(f"{class_name:15s}: AUC-ROC = {auc_score:.4f}")
        except Exception:
            print(f"{class_name:15s}: AUC-ROC = N/A (classe insuffisante)")

    mean_auc = np.mean(auc_scores) if auc_scores else 0
    print(f"\nMoyenne AUC-ROC: {mean_auc:.4f}")

    return accuracy, macro_f1, mean_auc


def analyze_errors(y_test, y_test_pred, le, augmenter, X_test, device):
    errors_indices = np.where(y_test != y_test_pred)[0]

    print(f"Nombre total d'erreurs : {len(errors_indices)}")

    augmenter.eval()

    for idx in errors_indices[:]:
        plt.figure(figsize=(10, 4))
        
        raw_audio = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                spec_tensor = augmenter(raw_audio, augment=False)
                
        spec_numpy = spec_tensor.squeeze().cpu().numpy()
        plt.imshow(spec_numpy, aspect='auto', origin='lower', cmap='viridis')
        
        real_class = le.classes_[y_test[idx]]
        pred_class = le.classes_[y_test_pred[idx]]
        
        plt.title(f"ERREUR - Réel: {real_class} | Prédit: {pred_class}")
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('Temps')
        plt.ylabel('Fréquence (Bandes Mel)')
        plt.tight_layout()
        plt.show()


def run_model(nom_model, directory, device, afficher_resultats=True):

    print(f"Travail sur modèle {nom_model} ...")
    
    X_train, X_features_train, y_train, X_val, X_features_val, y_val, X_test, X_features_test, y_test, le = preparation_donnees(directory)

    if nom_model == "ConvNeXt_with_features":
        train_loader, val_loader, augmenter = ConvNeXt_with_features.load_data(X_train, X_features_train, y_train, X_val, X_features_val, y_val, device)
        
        model, criterion, scaler, phases, best_acc, history = ConvNeXt_with_features.model_setup(X_features_train, le, device)
        model, history = ConvNeXt_with_features.train_model(model, train_loader, val_loader, augmenter, criterion, scaler, phases, best_acc, history, device)

        time.sleep(5)
        y_test_pred, y_test_proba = ConvNeXt_with_features.get_predictions(model, X_test, X_features_test, y_test, augmenter, device)
        accuracy, macro_f1, auc_scores = evaluate_model(y_test, y_test_pred, y_test_proba, le, afficher_resultats)
        #analyze_errors(y_test, y_test_pred, le, augmenter, X_test, device)
        return model, history, accuracy, macro_f1, auc_scores
    
    elif nom_model == "ConvNeXt":
        train_loader, val_loader, augmenter = ConvNeXt.load_data(X_train, y_train, X_val, y_val, device)
        
        model, criterion, scaler, phases, best_acc, history = ConvNeXt.model_setup(le, device)
        model, history = ConvNeXt.train_model(model, train_loader, val_loader, augmenter, criterion, scaler, phases, best_acc, history, device)
        
        time.sleep(5)

        y_test_pred, y_test_proba = ConvNeXt.get_predictions(model, X_test, y_test, augmenter, device)
        accuracy, macro_f1, auc_scores = evaluate_model(y_test, y_test_pred, y_test_proba, le, afficher_resultats)
        #analyze_errors(y_test, y_test_pred, le, augmenter, X_test, device)
        return model, history, accuracy, macro_f1, auc_scores

    elif nom_model == "CNN":
        train_loader, val_loader, augmenter = CNN.load_data(X_train, y_train, X_val, y_val, device)
        
        model, criterion, scaler, best_acc, history = CNN.model_setup(le, device, y_train)
        model, history = CNN.train_model(model, train_loader, val_loader, augmenter, criterion, scaler, best_acc, history, device)
        
        time.sleep(5)

        y_test_pred, y_test_proba = CNN.get_predictions(model, X_test, y_test, augmenter, device)
        accuracy, macro_f1, auc_scores = evaluate_model(y_test, y_test_pred, y_test_proba, le, afficher_resultats)
        #analyze_errors(y_test, y_test_pred, le, augmenter, X_test, device)
        return model, history, accuracy, macro_f1, auc_scores

    else:
        return None, None, None, None, None


if __name__ == "__main__":

    directory = "C:/Users/lrozier/Documents/UQAC/respiratory-disease-detection/data/processed/"

    # Configuration RTX 4070 : Tensor Cores (TF32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Propulsé par : {torch.cuda.get_device_name(0)}\n")

    choix_model = ["CNN", "ConvNeXt", "ConvNeXt_with_features"]
    indice_choix = int(input(f"Choisissez un modèle à entraîner/tester : \n\n\t0 -> CNN \n\t1 -> ConvNeXt \n\t2 -> ConvNeXt_with_features\n\nVotre choix : "))

    while indice_choix not in range(len(choix_model)):
        indice_choix = int(input(f"\nChoix invalide.\n\n Votre choix : "))


    model, history, accuracy, macro_f1, auc_scores = run_model(choix_model[indice_choix], directory, device)