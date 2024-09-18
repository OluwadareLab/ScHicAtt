import logging

def setup_logger(log_file):
    """
    Set up a logger that writes to a file.
    
    Args:
        log_file: Path to the log file.
    
    Returns:
        A logger instance that logs messages to the specified file.
    """
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')
    logger = logging.getLogger()
    return logger

def log_training_progress(logger, epoch, loss, psnr=None, ssim=None):
    """
    Log training progress information such as epoch, loss, and optional metrics.
    
    Args:
        logger: The logger instance used to log the information.
        epoch: Current epoch.
        loss: Loss value for the current epoch.
        psnr: (Optional) PSNR value for the current epoch.
        ssim: (Optional) SSIM value for the current epoch.
    """
    message = f"Epoch {epoch}, Loss: {loss:.4f}"
    
    if psnr is not None:
        message += f", PSNR: {psnr:.4f}"
    
    if ssim is not None:
        message += f", SSIM: {ssim:.4f}"
    
    logger.info(message)
