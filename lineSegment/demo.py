import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import UNet
from PIL import Image
import os
from losses import weighted_loss

in_channels = 3
out_channels = 1
image_path = '111.jpg'

model = UNet(in_channels=in_channels, out_channels=out_channels)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

input_image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
])
image = transform(input_image).unsqueeze(0)

pred = model(image)
pred = torch.sigmoid(pred)
# pred = pred.expand(-1, 3, -1, -1)
print(pred.squeeze(0).shape)
pred_mask = transforms.ToPILImage()(pred.squeeze(0))
pred_mask.save('res_00004.png',  'png')
print(pred_mask.size)
