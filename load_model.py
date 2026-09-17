import torch
from model import Autoencoder

def load_model(file_path):
    """
    Load the pre-trained Autoencoder model from a specified file path.

    Args:
        file_path (str): Path to the .pth file containing the model weights.

    Returns:
        Autoencoder: The loaded Autoencoder model.
    """
    input_dim = 12  # Based on the processed MFCCs shape (n_mfcc - 1)
    latent_dim = 32  # Assuming this was the latent dimension used during training
    model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim)
    try:
        state_dict = torch.load(file_path, map_location = torch.device("cpu"))
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model weights from {file_path}")
    except FileNotFoundError:
        print("Error: 'best_autoencoder.pth' not found. Please upload the file.")
        return None
    except RuntimeError as e:
        print(f"Error loading state dictionary: {e}")
        print("Please ensure the model architecture matches the saved state dictionary.")
        return None

    # Set the model to evaluation mode
    model.eval()
    print("Model loaded and set to evaluation mode.")
    return model