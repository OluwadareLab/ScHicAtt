import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loss_functions import mse_loss, PerceptualLoss, total_variation_loss, adversarial_loss
from model import Generator, Discriminator  # Assuming Generator and Discriminator are defined in model.py
from dataset import HiCDataset  # Assuming your dataset class is defined in dataset.py

# Composite Loss Function
def composite_loss(output, target, fake_data, discriminator, alpha=0.5, beta=0.3, gamma=0.1, delta=0.1):
    # Individual loss components
    mse = mse_loss(output, target)
    perceptual = PerceptualLoss()(output, target)
    tv = total_variation_loss(output)
    adv = adversarial_loss(discriminator, fake_data)

    # Weighted sum of losses
    total_loss = alpha * mse + beta * perceptual + gamma * tv + delta * adv
    return total_loss

# Training Function
def train_composite_loss(
    generator,
    discriminator,
    train_loader,
    val_loader,
    optimizer_g,
    optimizer_d,
    num_epochs,
    device,
    alpha=0.5,
    beta=0.3,
    gamma=0.1,
    delta=0.1
):
    generator.to(device)
    discriminator.to(device)
    perceptual_loss = PerceptualLoss().to(device)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        for i, (low_res, high_res) in enumerate(train_loader):
            low_res, high_res = low_res.to(device), high_res.to(device)

            # Generator forward pass
            fake_high_res = generator(low_res)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_loss = torch.mean((discriminator(high_res) - 1) ** 2)
            fake_loss = torch.mean(discriminator(fake_high_res.detach()) ** 2)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            g_loss = composite_loss(
                fake_high_res, high_res, fake_high_res, discriminator,
                alpha, beta, gamma, delta
            )
            g_loss.backward()
            optimizer_g.step()

            if i % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                    f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
                )

        # Validation Loop
        generator.eval()
        with torch.no_grad():
            val_loss = 0
            for low_res, high_res in val_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)
                fake_high_res = generator(low_res)
                val_loss += mse_loss(fake_high_res, high_res).item()

            val_loss /= len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_loss:.4f}")

# Main Script
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 50

    # Load Dataset
    train_dataset = HiCDataset("data/train")
    val_dataset = HiCDataset("data/val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Models
    generator = Generator()
    discriminator = Discriminator()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Train the model
    train_composite_loss(
        generator, discriminator,
        train_loader, val_loader,
        optimizer_g, optimizer_d,
        num_epochs, device
    )

if __name__ == "__main__":
    main()
