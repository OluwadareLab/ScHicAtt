class Config:
    """
    Configuration class for storing hyperparameters and file paths.
    Modify these values according to your experiment settings.
    """
    # Training parameters
    batch_size = 8
    epochs = 300
    learning_rate = 0.001
    attention_type = "self"  # Choices: "self", "local", "global", "dynamic"
    
    # Model saving path
    model_save_path = "saved_models/schicatt_model.pth"
    
    # Data file paths (modify these with actual paths)
    data_file = "path/to/data.npz"
    target_file = "path/to/target.npz"

    # Loss weights 
    loss_weights = {
        'mse': 1.0,         # Weight for MSE Loss
        'perceptual': 0.5,  # Weight for Perceptual Loss
        'tv': 0.01,         # Weight for Total Variation Loss
        'adversarial': 0.1  # Weight for Adversarial Loss
    }
