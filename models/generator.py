import torch.nn as nn
import torch
from .config import image_size


class Generator(nn.Module):
    def __init__(self, latent_dim=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),  # 节省内存
            nn.Linear(128, 256),
            torch.nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, image_size[0] * image_size[1] * image_size[2]),
            nn.Tanh(),
        )

    def forward(self, x):
        # shape of x: [batch_size, 1 ,28, 28]
        out = self.model(x)
        return out.view(out.size(0), *image_size)
