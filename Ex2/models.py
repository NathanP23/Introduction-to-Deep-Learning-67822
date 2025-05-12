import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleConvAutoencoder(nn.Module):
    def __init__(self, latent_dim, input_shape=(1, 28, 28)):
        super(FlexibleConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.input_channels = input_shape[0]
        
        # Encoder layers
        self.enc_conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        
        # Calculate the size of the flattened features
        self.flatten_size = self._get_flatten_size()
        
        # FC layer to latent dimension
        self.enc_fc = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder layers
        self.dec_fc = nn.Linear(latent_dim, self.flatten_size)
        
        # Transposed convolutions for upsampling
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0)
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(32, self.input_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def _get_flatten_size(self):
        # Helper function to calculate the flattened size after convolutions
        x = torch.zeros(1, *self.input_shape)
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        return x.numel()
    
    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.enc_fc(x)
        return x
    
    def decode(self, z):
        z = self.dec_fc(z)
        z = z.view(z.size(0), 128, 3, 3)  # Reshape to match the features before flattening
        z = F.relu(self.dec_conv1(z))
        z = F.relu(self.dec_conv2(z))
        z = torch.sigmoid(self.dec_conv3(z))  # Sigmoid to get values between 0 and 1
        return z
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

