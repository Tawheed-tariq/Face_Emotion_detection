import torch
import torch.nn as nn
from torchvision.models import resnet50

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(EmotionResNet, self).__init__()
        self.base_model = resnet50(pretrained=pretrained)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
