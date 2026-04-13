import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        ### FILL: add more CONV Layers
        # No BatchNorm on first encoder layer (standard practice for pix2pix-style networks)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),  # 512+512 skip
            nn.BatchNorm2d(256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),  # 256+256 skip
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),  # 128+128 skip
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),  # 64+64 skip
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        e1 = self.conv1(x)    # (B, 64, 128, 128)
        e2 = self.conv2(e1)   # (B, 128, 64, 64)
        e3 = self.conv3(e2)   # (B, 256, 32, 32)
        e4 = self.conv4(e3)   # (B, 512, 16, 16)
        e5 = self.conv5(e4)   # (B, 512, 8, 8)

        # Decoder forward pass with skip connections
        d5 = self.deconv5(e5)                           # (B, 512, 16, 16)
        d4 = self.deconv4(torch.cat([d5, e4], dim=1))   # (B, 256, 32, 32)
        d3 = self.deconv3(torch.cat([d4, e3], dim=1))   # (B, 128, 64, 64)
        d2 = self.deconv2(torch.cat([d3, e2], dim=1))   # (B, 64, 128, 128)
        output = self.deconv1(torch.cat([d2, e1], dim=1))  # (B, 3, 256, 256)

        return output
    