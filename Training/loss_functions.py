import torch
import torch.nn.functional as F
from torchvision import models

# Mean Squared Error (MSE) Loss
def mse_loss(output, target):
    return F.mse_loss(output, target)

# Perceptual Loss (based on VGG features)
class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = vgg[:16].eval()  # Use the first few layers for feature extraction
        for param in self.vgg_layers.parameters():
            param.requires_grad = False  # Fix VGG parameters

    def forward(self, x, y):
        x_vgg = self.vgg_layers(x)
        y_vgg = self.vgg_layers(y)
        return F.mse_loss(x_vgg, y_vgg)

# Total Variation (TV) Loss
def total_variation_loss(x):
    h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return h_tv + w_tv


def adversarial_loss(discriminator, fake_data):
    """
    Compute the adversarial loss for the generator.
    
    Args:
        discriminator: The discriminator model.
        fake_data: Generated data from the generator.
    
    Returns:
        Loss value (real or fake classification score).
    """
    fake_scores = discriminator(fake_data)
    return 1 - torch.mean(fake_scores)


