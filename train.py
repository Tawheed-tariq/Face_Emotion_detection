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


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet_emotion import EmotionResNet
from config import load_config
from utils.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load config
cfg = load_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
transform = transforms.Compose([
    transforms.Resize((cfg['training']['image_size'], cfg['training']['image_size'])),
    transforms.Grayscale(num_output_channels=3) if cfg['training']['grayscale'] else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Datasets and DataLoaders
train_dataset = datasets.ImageFolder(cfg['dataset']['train_path'], transform=transform)
val_dataset = datasets.ImageFolder(cfg['dataset']['val_path'], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

# Model, loss, optimizer
model = EmotionResNet(num_classes=cfg['training']['num_classes'], pretrained=cfg['training']['pretrained']).to(device)

# Load from checkpoint if specified
if cfg.get('ckpt_path'):
    print(f"Loading model from checkpoint: {cfg['ckpt_path']}")
    model.load_state_dict(torch.load(cfg['ckpt_path'], map_location=device))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

# Scheduler
scheduler = None
if cfg['scheduler']['use_scheduler']:
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=cfg['scheduler']['mode'],
        factor=cfg['scheduler']['factor'],
        patience=cfg['scheduler']['patience']
    )

# Early Stopping
early_stopping = None
if cfg['early_stopping']['use_early_stopping']:
    early_stopping = EarlyStopping(patience=cfg['early_stopping']['patience'], verbose=True)

# Training loop
for epoch in range(cfg['training']['epochs']):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    with open(cfg['training']['log_file'], 'a') as log_file:
        log_file.write(f"Epoch_{epoch+1}_trainLoss_{avg_train_loss:.4f}_valLoss_{avg_val_loss:.4f}\n")
    # Print progress
    print(f"Epoch_{epoch+1}_trainLoss_{avg_train_loss:.4f}_valLoss_{avg_val_loss:.4f}\n")

    # Scheduler step
    if scheduler:
        scheduler.step(avg_val_loss)

    # Save model
    save_path = cfg['training']['save_path'].format(epoch=epoch + 1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    # Early stopping check
    if early_stopping:
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
