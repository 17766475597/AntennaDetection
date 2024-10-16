import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Contracting path (Encoder)
        self.conv1 = self.double_conv(in_channels, 64)
        self.conv2 = self.double_conv(64, 128)
        self.conv3 = self.double_conv(128, 256)
        self.conv4 = self.double_conv(256, 512)
        self.conv5 = self.double_conv(512, 1024)

        # Expansive path (Decoder)
        self.upconv4 = self.upconv(1024, 512)
        self.conv6 = self.double_conv(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.conv7 = self.double_conv(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.conv8 = self.double_conv(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.conv9 = self.double_conv(128, 64)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(F.max_pool2d(x1, 2))
        x3 = self.conv3(F.max_pool2d(x2, 2))
        x4 = self.conv4(F.max_pool2d(x3, 2))
        x5 = self.conv5(F.max_pool2d(x4, 2))

        # Decoder
        x = self.upconv4(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv6(x)

        x = self.upconv3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv7(x)

        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv8(x)

        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv9(x)

        # Output
        return self.out_conv(x)

# # Example usage:
# # Define the model
# model = UNet(in_channels=3, out_channels=1)

# # Example input (batch size 1, 3 channels, 256x256 image)
# input_tensor = torch.randn(1, 3, 256, 256)

# # Forward pass
# output = model(input_tensor)

# print(output.shape)  # should be [1, 1, 256, 256]
