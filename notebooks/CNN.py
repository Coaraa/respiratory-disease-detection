import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import DataLoader, Dataset

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


class CNNBaseline(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,   32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x): return self.classifier(self.features(x))

def load_data(X_train, y_train, X_val, y_val, device):
    augmenter = GPUAugmenter(sr=22050).to(device)
    train_loader = DataLoader(Dataset(X_train, y_train), batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(Dataset(X_val, y_val), batch_size=64, pin_memory=True)
    return train_loader, val_loader, augmenter

def model_setup(le, device, y_train):    

    counts  = np.bincount(y_train)
    weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    weights = weights / weights.sum() * len(le.classes_)

    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler('cuda')
    model = CNNBaseline(num_classes=len(le.classes_)).to(device)

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    return model, criterion, scaler, best_acc, history

def train_model(model, train_loader, val_loader, augmenter, criterion, scaler, best_acc, history, device):

        
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=1e-3, weight_decay=1e-4)
    nb_epochs = 80

    for epoch in range(nb_epochs): 
        model.train()
        total_loss = 0
        for audios,labels in train_loader:
            audios, labels = audios.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                # Augmentation GPU
                specs = augmenter(audios, augment=True)
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
        
        print(f"Epoch {epoch+1}/{nb_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.2%}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model_cnn.pth")
    
    return model, history

def get_predictions(model, X_test, y_test, augmenter, device):
    test_loader = DataLoader(Dataset(X_test, y_test), batch_size=64, shuffle=False, pin_memory=True)

    if model is None:
        raise ValueError("Le modèle doit être fourni en paramètre.")

    model.eval()
    y_test_pred = []
    y_test_proba = []

    with torch.no_grad():
        for audios, labels in test_loader:
            audios = audios.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                # Transformation en Spectrogramme sur GPU sans augmentation
                specs = augmenter(audios, augment=False)
                
                outputs = model(specs)
                
            proba = torch.nn.functional.softmax(outputs, dim=1)
            y_test_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_test_proba.extend(proba.cpu().numpy())

    y_test_pred = np.array(y_test_pred)
    y_test_proba = np.array(y_test_proba)

    return y_test_pred, y_test_proba




