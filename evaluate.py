from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet_emotion import EmotionResNet
from config import load_config

cfg = load_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = EmotionResNet(num_classes=cfg['training']['num_classes'])
model.load_state_dict(torch.load("checkpoints/resnet_emotion_epoch20.pth"))
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
test_dataset = datasets.ImageFolder(cfg['dataset']['fer2013_test_path'], transform=transform)
test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

# Evaluation
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_dataset.classes))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
