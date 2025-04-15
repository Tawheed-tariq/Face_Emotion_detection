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
from tqdm import tqdm
import datetime

# Load config
cfg = load_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create timestamped output directory
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join("output", timestamp)
os.makedirs(output_dir, exist_ok=True)
checkpoints_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoints_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "log.txt")

# Update config paths with new output directory
save_path_template = os.path.join(checkpoints_dir, str(cfg['training']['checkpoint_pattern']) if cfg['training']['checkpoint_pattern'] else "epoch={epoch}-val_acc={val_accuracy}-val_loss={val_loss}.pth")

print(f"Output directory created at: {output_dir}")
print(f"Checkpoints will be saved to: {checkpoints_dir}")
print(f"Log file will be saved at: {log_file_path}")

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

# Initialize starting epoch
start_epoch = 0

# Load from checkpoint if specified
if cfg['ckpt_path'] is not None:
    print(f"Loading model from checkpoint: {cfg['ckpt_path']}")
    checkpoint = torch.load(cfg['ckpt_path'], map_location=device)
    
    # Check if checkpoint is just model state dict or a dictionary with more info
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        print(f"Resuming from epoch {start_epoch}")
        
        # Also load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'output_dir' in checkpoint:
            output_dir = checkpoint['output_dir']
            checkpoints_dir = os.path.join(output_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            print(f"Output directory restored from checkpoint: {output_dir}")
    else:
        # Just load the model weights if only state_dict was saved
        model.load_state_dict(checkpoint)
        print("Loaded only model weights, starting from epoch 1")

# If optimizer wasn't loaded from checkpoint, create a new one
if 'optimizer' not in locals():
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

criterion = nn.CrossEntropyLoss()

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

# Write training configuration to log file
with open(log_file_path, 'w') as log_file:
    log_file.write(f"Training started at: {timestamp}\n")
    log_file.write(f"Model: EmotionResNet\n")
    log_file.write(f"Number of classes: {cfg['training']['num_classes']}\n")
    log_file.write(f"Batch size: {cfg['training']['batch_size']}\n")
    log_file.write(f"Learning rate: {cfg['training']['learning_rate']}\n")
    log_file.write(f"Image size: {cfg['training']['image_size']}\n")
    log_file.write(f"Grayscale: {cfg['training']['grayscale']}\n")
    log_file.write(f"Pretrained: {cfg['training']['pretrained']}\n")
    log_file.write(f"Device: {device}\n")
    log_file.write("\n--- Training Progress ---\n")
log_file_path = os.path.join(output_dir, "log.txt")
# Training loop
for epoch in range(start_epoch, cfg['training']['epochs']):
    print(f"Epoch {epoch + 1}/{cfg['training']['epochs']}")
    model.train()
    running_loss = 0.0

    train_loop = tqdm(train_loader, desc="Training")
    for images, labels in train_loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        train_loop.set_postfix(loss=loss.item())

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    val_loop = tqdm(val_loader, desc="Validating")
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            val_loop.set_postfix(val_loss=loss.item(), accuracy=f"{100 * correct / total:.2f}%")

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Epoch_{epoch+1}_trainLoss_{avg_train_loss:.4f}_valLoss_{avg_val_loss:.4f}_valAcc_{val_accuracy:.2f}%\n")
    
    # Print progress
    print(f"Epoch_{epoch+1}_trainLoss_{avg_train_loss:.4f}_valLoss_{avg_val_loss:.4f}_valAcc_{val_accuracy:.2f}%\n")

    # Scheduler step
    if scheduler:
        scheduler.step(avg_val_loss)

    # Save model with more information for resuming
    save_path = save_path_template.format(epoch=epoch + 1, val_accuracy=val_accuracy, val_loss=avg_val_loss)
    
    # Save model with epoch info for resuming later
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'train_loss': avg_train_loss,
        'val_accuracy': val_accuracy,
        'output_dir': output_dir
    }
    torch.save(checkpoint, save_path)

    # Save the best model separately
    if early_stopping and early_stopping.counter == 0:  # This is a new best model
        best_model_path = os.path.join(checkpoints_dir, "best_model.pth")
        torch.save(checkpoint, best_model_path)
        print(f"Saved new best model with validation loss: {avg_val_loss:.4f}")

    # Early stopping check
    if early_stopping:
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

# Final message
print(f"Training completed. All outputs saved to: {output_dir}")