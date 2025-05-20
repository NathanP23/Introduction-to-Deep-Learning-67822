from torch import nn
from mnistlib.blocks import convolution_block_relu, deconvolution_block_relu

# mnistlib/models.py
class Encoder(nn.Module):
    def __init__(self, base_channel_count: int, latent_dimension: int):
        """
        Initialize the encoder module with specified channel counts and latent dimension
        
        Args:
            base_channel_count (int): Base number of channels for the convolutional layers
            latent_dimension (int): Dimension of the latent space representation
            
        Returns:
            None
        """
        super().__init__()
        channels_level_1, channels_level_2, channels_level_3 = base_channel_count, base_channel_count*2, base_channel_count*4
        self.backbone = nn.Sequential(
            convolution_block_relu(1, channels_level_1),   # 28→14
            convolution_block_relu(channels_level_1, channels_level_2),   # 14→7
            convolution_block_relu(channels_level_2, channels_level_3),   # 7→4
        )
        self.linear_projection_layer = nn.Linear(channels_level_3*4*4, latent_dimension)

    def forward(self, input_images):
        """
        Forward pass of the encoder
        
        Args:
            input_images (torch.Tensor): Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            torch.Tensor: Latent representation of shape [batch_size, latent_dimension]
        """
        feature_maps = self.backbone(input_images).flatten(1)
        latent_vectors = self.linear_projection_layer(feature_maps)
        return latent_vectors

class Decoder(nn.Module):
    def __init__(self, base_channel_count: int, latent_dimension: int):
        """
        Initialize the decoder module with specified channel counts and latent dimension
        
        Args:
            base_channel_count (int): Base number of channels for the deconvolutional layers
            latent_dimension (int): Dimension of the latent space representation
            
        Returns:
            None
        """
        super().__init__()
        channels_level_1, channels_level_2, channels_level_3 = base_channel_count, base_channel_count*2, base_channel_count*4
        self.linear_projection_layer = nn.Linear(latent_dimension, channels_level_3*4*4)
        self.upsampling_stack = nn.Sequential(
            deconvolution_block_relu(channels_level_3, channels_level_2, output_padding=0),   # 4→7
            deconvolution_block_relu(channels_level_2, channels_level_1, output_padding=1),   # 7→14
            nn.ConvTranspose2d(channels_level_1, 1, 3, 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, latent_vectors):
        """
        Forward pass of the decoder
        
        Args:
            latent_vectors (torch.Tensor): Latent representation of shape [batch_size, latent_dimension]
            
        Returns:
            torch.Tensor: Reconstructed image of shape [batch_size, 1, 28, 28]
        """
        initial_feature_maps = self.linear_projection_layer(latent_vectors).view(latent_vectors.size(0), -1, 4, 4)
        reconstructed_images = self.upsampling_stack(initial_feature_maps)
        return reconstructed_images

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, base_channel_count: int, latent_dimension: int):
        """
        Initialize the convolutional autoencoder with specified channel counts and latent dimension
        
        Args:
            base_channel_count (int): Base number of channels for the convolutional layers
            latent_dimension (int): Dimension of the latent space representation
            
        Returns:
            None
        """
        super().__init__()
        self.encoder = Encoder(base_channel_count, latent_dimension)
        self.decoder = Decoder(base_channel_count, latent_dimension)
    
    def forward(self, input_images):
        """
        Forward pass of the autoencoder
        
        Args:
            input_images (torch.Tensor): Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            torch.Tensor: Reconstructed image of shape [batch_size, 1, 28, 28]
        """
        latent_vectors = self.encoder(input_images)
        reconstructed_images = self.decoder(latent_vectors)
        return reconstructed_images

class MLPClassifier(nn.Module):
    def __init__(self, base_channel_count: int, latent_dimension: int):
        """
        Initialize the MLP classifier with specified channel counts and latent dimension
        
        Args:
            base_channel_count (int): Base number of channels for the convolutional layers
            latent_dimension (int): Dimension of the latent space representation
            
        Returns:
            None
        """
        super().__init__()
        self.encoder = Encoder(base_channel_count, latent_dimension)
        self.classification_layer = nn.Linear(latent_dimension, 10)

    def forward(self, input_images):
        """
        Forward pass of the MLP classifier
        
        Args:
            input_images (torch.Tensor): Input tensor of shape [batch_size, 1, 28, 28]
            
        Returns:
            torch.Tensor: Class logits of shape [batch_size, 10]
        """
        latent_vectors = self.encoder(input_images)
        class_logits = self.classification_layer(latent_vectors)
        return class_logits