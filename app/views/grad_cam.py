import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models

# Load an image and preprocess it
def preprocess_image(img_array):
    img = cv2.resize(img_array, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0)
    img = transforms.Normalize(mean=[0.5], std=[0.5])(img)
    img = img.unsqueeze(0)
    
    return img


def get_conv_layer(model, conv_layer_name):
    for name, layer in model.named_modules():
        if name == conv_layer_name:
            return layer
    raise ValueError(f"Layer '{conv_layer_name}' not found in the model.")

# Function to generate Grad-CAM heatmap
def compute_gradcam(model, img_tensor, class_index, conv_layer_name="cnn.features.7"):
    # Ensure gradients are tracked
    img_tensor.requires_grad_(True)
    
    # 1. Get the target layer
    target_layer = dict([*model.named_modules()])[conv_layer_name]

    # 2. Hook to capture activations and gradients
    activations = []
    gradients = []

    def save_activation(module, input, output):
        activations.append(output)
    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    a_hook = target_layer.register_forward_hook(save_activation)
    g_hook = target_layer.register_full_backward_hook(save_gradient)

    # 3. Forward and Backward pass
    model.zero_grad()
    output = model(img_tensor)
    loss = output[0, class_index]
    loss.backward()

    # 4. Process Heatmap
    # ConvNeXt outputs (B, C, H, W). We take [0] for batch
    grad = gradients[0][0].cpu().data.numpy()
    act = activations[0][0].cpu().data.numpy()
    
    weights = np.mean(grad, axis=(1, 2))
    heatmap = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        heatmap += w * act[i, :, :]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    # Clean up
    a_hook.remove()
    g_hook.remove()
    
    return heatmap

def overlay_heatmap(img_array, heatmap, alpha=0.3):
    """
    Superpose la heatmap sur le spectrogramme original.
    img_array: Le spectrogramme original (uint8, niveaux de gris).
    heatmap: La carte d'activation générée par compute_gradcam.
    """
    # 1. Redimensionner la heatmap à la taille exacte du spectrogramme initial (Largeur, Hauteur)
    # OpenCV utilise (width, height)
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    
    # 2. Convertir la heatmap en couleurs (JET : bleu vers rouge)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # 3. Convertir le spectrogramme original (gris) en BGR pour pouvoir ajouter de la couleur
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    
    # 4. Fusionner les deux images
    superimposed_img = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_color, alpha, 0)
    
    # 5. Convertir en RGB pour l'affichage correct dans Streamlit
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)