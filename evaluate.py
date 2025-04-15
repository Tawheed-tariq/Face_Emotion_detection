#!/usr/bin/env python3
# Face Emotion Recognition Model
# Copyright 2025 Tavaheed Tariq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet_emotion import EmotionResNet
from config import load_config
from util import print_metrics
from tqdm import tqdm

checkpoint_path = "/home/tawheed/HumanitiesProject/Face_Emotion_detection/output/2025-04-15_08-20-43/checkpoints/best_model.pth"  # Update with your checkpoint path
reslts_path = "/home/tawheed/HumanitiesProject/Face_Emotion_detection/output/2025-04-15_08-20-43/results.txt"  # Update with your results path

cfg = load_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = EmotionResNet(num_classes=cfg['training']['num_classes'])
checkpoint = torch.load(checkpoint_path)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((cfg['training']['image_size'], cfg['training']['image_size'])),
    transforms.Grayscale(num_output_channels=3) if cfg['training']['grayscale'] else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Test set
test_dataset = datasets.ImageFolder(cfg['dataset']['test_path'], transform=transform)
test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

# Evaluation
y_true, y_pred = [], []
with torch.no_grad():
    test_loop = tqdm(test_loader, desc="Evaluating", leave=False)
    for images, labels in test_loop:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        test_loop.set_postfix({"Accuracy": f"{(predicted == labels).float().mean():.4f}"})

with open(reslts_path, "w") as f:
    f.write("Classification Report:\n")
    report = classification_report(y_true, y_pred, target_names=test_dataset.classes)
    f.write(f"{report}\n")
    f.write("\nConfusion Matrix:\n")
    cm = confusion_matrix(y_true, y_pred)
    f.write(str(cm))
    f.write("\n")
    f.write("Confusion Matrix (Normalized):\n")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    f.write(str(cm_normalized))
    f.write("\n")
    f.write("Metrics:\n")
    f.write(str(print_metrics(y_true, y_pred)))
