# mnistlib/ae.py
import torch.nn as nn
from .blocks import conv_bn_relu, deconv_bn_relu

class Encoder(nn.Module):
    def __init__(self, base_c: int, d_latent: int):
        super().__init__()
        c1,c2,c3 = base_c, base_c*2, base_c*4
        self.backbone = nn.Sequential(
            conv_bn_relu(1 , c1),   # 28→14
            conv_bn_relu(c1, c2),   # 14→7
            conv_bn_relu(c2, c3),   # 7 →4
        )
        self.fc = nn.Linear(c3*4*4, d_latent)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, base_c: int, d_latent: int):
        super().__init__()
        c1,c2,c3 = base_c, base_c*2, base_c*4
        self.fc = nn.Linear(d_latent, c3*4*4)
        self.up = nn.Sequential(
            deconv_bn_relu(c3, c2, op=0),   # 4→7
            deconv_bn_relu(c2, c1, op=1),   # 7→14
            nn.ConvTranspose2d(c1, 1, 3, 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0), -1, 4, 4)
        return self.up(x)

class ConvAE(nn.Module):
    def __init__(self, base_c: int, d_latent: int):
        super().__init__()
        self.enc = Encoder(base_c, d_latent)
        self.dec = Decoder(base_c, d_latent)
    def forward(self, x):  return self.dec(self.enc(x))
