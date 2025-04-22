import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from models.resnet_emotion import EmotionResNet
import yaml
from config import load_config
import pandas as pd

# ----------- Load Config ------------
cfg = load_config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- Load Model ------------
model = EmotionResNet(num_classes=cfg['training']['num_classes'], pretrained=cfg['training']['pretrained']).to(device)
checkpoint = torch.load(cfg['test']['ckpt'], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ----------- Image Transform ------------
transform = transforms.Compose([
    transforms.Resize((cfg['training']['image_size'], cfg['training']['image_size'])),
    transforms.Grayscale(num_output_channels=3) if cfg['training']['grayscale'] else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# ----------- Prediction Folder ------------
img_dir = cfg['test']['img_dir']  # folder with images
class_names = cfg['dataset']['class_names']  # list of class names
df = pd.DataFrame(columns=['image_name'] + class_names + ['predicted_class'])


for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)
    if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

        pred_class = class_names[pred.item()]
        # conf_percent = confidence.item() * 100

        

        # Add a new row to the DataFrame
        df.loc[len(df)] = {
            'image_name': img_name,
            **{class_names[i]: round(probs[0][i].item(), 4) for i in range(len(class_names))},
            'predicted_class': pred_class
        }

df.to_csv(os.path.join(cfg['test']['output_dir'], 'predictions.csv'), index=False)
       
