import torch

def save_model(model, optimizer, epoch, save_path):
    """
    Save the model's state and optimizer's state.
    
    Args:
        model: The model to be saved.
        optimizer: The optimizer used for training the model.
        epoch: The current epoch number.
        save_path: The file path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved at {save_path}")

def load_model(model, optimizer, load_path):
    """
    Load a model's state and optimizer's state from a checkpoint file.
    
    Args:
        model: The model to load the state into.
        optimizer: The optimizer to load the state into.
        load_path: The file path from where to load the checkpoint.
    
    Returns:
        model: Model with loaded state.
        optimizer: Optimizer with loaded state.
        epoch: The epoch from which training was resumed.
    """
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {load_path}, resuming from epoch {epoch + 1}")
    return model, optimizer, epoch
