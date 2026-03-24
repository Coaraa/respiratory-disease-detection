import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_tiny

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

class Dataset(Dataset):
    def __init__(self, x_audio, y_labels):
        self.x_audio = x_audio
        self.y = y_labels
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        audio = torch.tensor(self.x_audio[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return audio, label


class RespiratoryModel(nn.Module):
    def __init__(self, num_classes=5, weights="DEFAULT"):
        super().__init__()
        self.cnn = convnext_tiny(weights=weights)
        
        # Adaptation 1 canal
        old_conv = self.cnn.features[0][0]
        self.cnn.features[0][0] = nn.Conv2d(1, old_conv.out_channels, kernel_size=4, stride=4)
        with torch.no_grad():
            self.cnn.features[0][0].weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        
        self.cnn.classifier = nn.Identity()

        # Adaptation de la sortie 5 classes
        self.classifier = nn.Sequential(
            nn.LayerNorm((768, 1, 1), eps=1e-6),
            nn.Flatten(1),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )


    def forward(self, spec):
        cnn_out = self.cnn(spec)
        out = self.classifier(cnn_out)
        return out


def load_data(X_train, y_train, X_val, y_val, device):
    augmenter = GPUAugmenter(sr=22050).to(device)
    train_loader = DataLoader(Dataset(X_train, y_train), batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(Dataset(X_val, y_val), batch_size=64, pin_memory=True)
    return train_loader, val_loader, augmenter


def model_setup(le, device):    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    model = RespiratoryModel(num_classes=len(le.classes_)).to(device)

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    phases = [{"name": "Warmup (Freeze)", "epochs": 10, "lr": 5e-4, "freeze": True},
              {"name": "Fine-Tuning (Unfreeze)", "epochs": 80, "lr": 5e-5, "freeze": False}]

    return model, criterion, scaler, phases, best_acc, history


def train_model(model, train_loader, val_loader, augmenter, criterion, scaler, phases, best_acc, history, device):
    for p_info in phases:
        print(f"\n>>> DÉBUT PHASE : {p_info['name']}")
        
        # Gestion du Freeze/Unfreeze
        for param in model.cnn.parameters():
            param.requires_grad = not p_info['freeze']
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=p_info['lr'], weight_decay=1e-4)

        for epoch in range(p_info['epochs']):
            model.train()
            total_loss = 0
            for audios,labels in train_loader:
                audios, labels = audios.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    # Augmentation GPU 
                    specs = augmenter(audios, augment=(not p_info['freeze']))
                    outputs = model(specs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            # Validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for audios, labels in val_loader:
                    audios, labels = audios.to(device), labels.to(device)
                    with torch.amp.autocast('cuda'):
                        specs = augmenter(audios, augment=False)
                        outputs = model(specs)
                    correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            
            acc = correct / total
            history["train_loss"].append(total_loss/len(train_loader))
            history["val_acc"].append(acc)
            
            print(f"Epoch {epoch+1}/{p_info['epochs']} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.2%}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "best_model_respiratory.pth")
    
    return model, history


def get_predictions(model, X_test, y_test, augmenter, device):
    test_loader = DataLoader(Dataset(X_test, y_test), batch_size=64, shuffle=False, pin_memory=True)

    if model is None:
        model = RespiratoryModel(num_classes=len(torch.le.classes_)).to(device)
        model.load_state_dict(torch.load("../models/best_model_respiratory.pth"))

    # Faire les prédictions sur le test set
    model.eval()
    y_test_pred = []
    y_test_proba = []

    with torch.no_grad():
        for audios, labels in test_loader:
            # On envoie l'audio brut 
            audios = audios.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                # Transformation en Spectrogramme sur GPU sans augmentation
                specs = augmenter(audios, augment=False)
                
                # Prédiction du modèle
                outputs = model(specs)
                
            proba = torch.nn.functional.softmax(outputs, dim=1)
            y_test_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_test_proba.extend(proba.cpu().numpy())

    y_test_pred = np.array(y_test_pred)
    y_test_proba = np.array(y_test_proba)

    return y_test_pred, y_test_proba

