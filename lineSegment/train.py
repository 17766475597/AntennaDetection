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

class CustomSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): 图像文件所在的目录，图像为 .jpg 文件。
            mask_dir (str): 掩码文件所在的目录，掩码为 .png 文件。
            transform (callable, optional): 对图像和掩码的变换操作。
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.mask_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        # print(img_filename, idx)
        mask_filename = self.mask_filenames[idx]

        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


def train_model(model, train_loader, criterion1, criterion2, optimizer, num_epochs=25, device='cuda'):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # print(images.shape, masks.shape)
            images, masks = images.to(device), masks.to(device)
            masks[masks < 0.5] = int(0)
            masks[masks >= 0.5] = int(1)

            outputs = model(images)
            loss1 = criterion1(outputs, masks)
            loss2 = criterion2(outputs, masks)
            loss = loss1 + 100*loss2
            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for images, masks in val_loader:
        #         images, masks = images.to(device), masks.to(device)

        #         outputs = model(images)
        #         loss = criterion(outputs, masks)
        #         val_loss += loss.item()

        # avg_val_loss = val_loss / len(val_loader)
        # print(f"Validation Loss: {avg_val_loss:.4f}")


num_epochs = 200
learning_rate = 0.00001
batch_size = 8
in_channels = 3 
out_channels = 1

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_image_paths = "all_data1/"
train_mask_paths = "all_data1/"

train_dataset = CustomSegmentationDataset(train_image_paths, train_mask_paths, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = UNet(in_channels=in_channels, out_channels=out_channels)
criterion1 = nn.BCEWithLogitsLoss()
criterion2 = weighted_loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_model(model, train_loader, criterion1, criterion2, optimizer, num_epochs=num_epochs, device='cuda')

torch.save(model.state_dict(), 'model_weights.pth')
