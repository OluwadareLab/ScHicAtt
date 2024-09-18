import sys
import torch
import torch.optim as optim
from models.generator import Generator
from data_loader import get_data_loader
from model_saver import save_model, load_model
from config import Config
from loss_functions import mse_loss, PerceptualLoss, total_variation_loss, adversarial_loss
from train_helpers import calculate_psnr
from logger import setup_logger, log_training_progress
from discriminator import Discriminator  # Import the discriminator model
import argparse

# Discriminator loss function
def discriminator_loss(discriminator, real_data, fake_data):
    real_scores = discriminator(real_data)
    fake_scores = discriminator(fake_data.detach())
    real_loss = torch.mean((real_scores - 1) ** 2)  # Real should be 1
    fake_loss = torch.mean(fake_scores ** 2)  # Fake should be 0
    return (real_loss + fake_loss) / 2

# Compute the total loss based on the individual losses and weights for the generator
def compute_total_loss(output, target, perceptual_loss, discriminator, loss_weights):
    mse = mse_loss(output, target) * loss_weights['mse']
    perceptual = perceptual_loss(output, target) * loss_weights['perceptual']
    tv = total_variation_loss(output) * loss_weights['tv']
    adv = adversarial_loss(discriminator, output) * loss_weights['adversarial']

    total_loss = mse + perceptual + tv + adv
    return total_loss

# Main training loop
def train_model(model, discriminator, data_loader, optimizer, optimizer_d, loss_weights, epochs, device, model_save_path, logger):
    perceptual_loss = PerceptualLoss().to(device)

    for epoch in range(epochs):
        model.train()
        total_g_loss = 0
        total_d_loss = 0
        psnr_total = 0

        for batch in data_loader:
            inputs, targets = batch['input'].to(device), batch['target'].to(device)

            # Generate output from generator
            outputs = model(inputs)

            # Train discriminator
            optimizer_d.zero_grad()
            d_loss = discriminator_loss(discriminator, targets, outputs)
            d_loss.backward()
            optimizer_d.step()
            total_d_loss += d_loss.item()

            # Train generator
            optimizer.zero_grad()
            g_loss = compute_total_loss(outputs, targets, perceptual_loss, discriminator, loss_weights)
            g_loss.backward()
            optimizer.step()
            total_g_loss += g_loss.item()

            # Calculate PSNR
            psnr_value = calculate_psnr(outputs, targets)
            psnr_total += psnr_value

        avg_g_loss = total_g_loss / len(data_loader)
        avg_d_loss = total_d_loss / len(data_loader)
        avg_psnr = psnr_total / len(data_loader)

        print(f"Epoch [{epoch + 1}/{epochs}], Generator Loss: {avg_g_loss:.4f}, Discriminator Loss: {avg_d_loss:.4f}, PSNR: {avg_psnr:.4f}")
        log_training_progress(logger, epoch + 1, avg_g_loss, avg_psnr)

        # Save the model at each epoch
        save_model(model, optimizer, epoch, model_save_path)

    print("Training complete. Model saved at", model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training process for ScHiCAtt.")
    parser.add_argument('-e', '--epoch', type=int, default=Config.epochs, help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=Config.batch_size, help='Batch size for training')
    parser.add_argument('-a', '--attention', type=str, choices=["self", "local", "global", "dynamic"], default=Config.attention_type, help="Type of attention mechanism to use")
    parser.add_argument('--load_model', type=str, help='Path to a saved model to resume training (optional)')
    args = parser.parse_args()

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logger
    logger = setup_logger("training_log.txt")

    # Instantiate the model and discriminator
    model = Generator(num_channels=64, attention_type=args.attention).to(device)
    discriminator = Discriminator(in_channels=1).to(device)  # Instantiate the discriminator

    # Define optimizers
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)  # For generator
    optimizer_d = optim.Adam(discriminator.parameters(), lr=Config.learning_rate)  # For discriminator

    # Loss weights (adjust as needed)
    loss_weights = Config.loss_weights

    # Load data
    data_loader = get_data_loader(data_file=Config.data_file, target_file=Config.target_file, batch_size=args.batch_size)

    # Optionally load a saved model to resume training
    start_epoch = 0
    if args.load_model:
        model, optimizer, start_epoch = load_model(model, optimizer, args.load_model)
        print(f"Resuming training from epoch {start_epoch + 1}")

    # Train the model
    train_model(model, discriminator, data_loader, optimizer, optimizer_d, loss_weights, epochs=args.epoch, device=device, model_save_path=Config.model_save_path, logger=logger)
