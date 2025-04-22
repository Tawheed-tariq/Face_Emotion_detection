import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from models.resnet_emotion import EmotionResNet
from config import load_config
import pandas as pd
import os
from datetime import datetime

cfg = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionResNet(num_classes=cfg['training']['num_classes'], pretrained=cfg['training']['pretrained']).to(device)
checkpoint = torch.load(cfg['test']['ckpt'], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((cfg['training']['image_size'], cfg['training']['image_size'])),
    transforms.Grayscale(num_output_channels=3) if cfg['training']['grayscale'] else transforms.Lambda(lambda x: x),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class_names = cfg['dataset']['class_names']

output_csv_path = os.path.join(cfg['test']['output_dir'], 'realtime_predictions.csv')
df = pd.DataFrame(columns=['timestamp'] + class_names + ['predicted_class'])

# ----------- Start Webcam ------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    pred_class = class_names[pred.item()]
    conf_percent = probs[0][pred.item()].item() * 100

    # Overlay prediction
    label = f"{pred_class}: {conf_percent:.1f}%"
    cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Real-Time Emotion Recognition", frame)

    # Log with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {'timestamp': timestamp,
           **{class_names[i]: round(probs[0][i].item(), 4) for i in range(len(class_names))},
           'predicted_class': pred_class}
    df.loc[len(df)] = row

    # Save CSV live (optional: can save only at the end instead)
    df.to_csv(output_csv_path, index=False)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
