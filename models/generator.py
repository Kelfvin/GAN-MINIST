import torch.nn as nn
import torch
from .config import image_size


class Generator(nn.Module):
    def __init__(self, latent_dim=64):
        super(Generator, self).__init__()

        self.fc = nn.Linear(latent_dim, 56 * 56)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self.conBr1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # (x-3+2)/1 + 1 = x 高宽不变
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conBr2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),  # 高宽不变
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conBr3 = nn.Sequential(
            nn.Conv2d(32, 1, 4, 2, 1),  # 高宽减半 (x-4+2)/2 + 1 = x/2
            nn.Tanh(),
        )

    def forward(self, x):
        # shape of x: [batch_size, 1 ,28, 28]
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.conBr1(x)
        x = self.conBr2(x)
        x = self.conBr3(x)
        return x
