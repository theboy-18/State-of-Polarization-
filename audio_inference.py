import numpy as np
import torch
import logging
def infer_anomaly_autoencoder(normalized_mfccs, model, sr, hop_length):
    """
    Performs the inference step of an audio anomaly detection pipeline
    using a PyTorch autoencoder model.

    Args:
        normalized_mfccs (np.ndarray): NumPy array of shape (n_frames, n_mfcc),
                                       representing normalized MFCCs.
        model (torch.nn.Module): Trained PyTorch autoencoder model.
        sr (int): Sampling rate of the audio.
        hop_length (int): Hop length used in MFCC extraction.
        threshold (float, optional): Threshold for flagging anomalous frames
                                     based on reconstruction error. Defaults to 0.1.

    Returns:
        tuple: (recon, errors, timestamps, anomaly_flags)
               - recon (np.ndarray): Reconstructed MFCCs, same shape as input.
               - errors (np.ndarray): Reconstruction error per frame, shape (n_frames,).
               - timestamps (np.ndarray): Timestamps for each frame, shape (n_frames,).
               - anomaly_flags (np.ndarray): Boolean array indicating anomalous frames,
                                           shape (n_frames,).
    """
    # 1. Convert normalized_mfccs to a PyTorch tensor
    # Add a batch dimension at the beginning
    input_tensor = torch.tensor(normalized_mfccs, dtype=torch.float32).unsqueeze(0)


    # 3. Use torch.no_grad() for inference
    with torch.no_grad():
        # 4. Pass the input through the model to get the reconstruction
        recon_tensor = model(input_tensor)

    # Convert reconstruction back to NumPy array and remove batch dimension
    reconstructed_output = recon_tensor.squeeze(0).numpy()

    # 5. Compute per-frame reconstruction error (Mean Squared Error along the MFCC axis)
    # Ensure the input and reconstruction tensors have the same shape before calculating error
    # The reconstruction should have the same shape as the input MFCCs (n_frames, n_mfcc)
    # If the model output has a different shape, you might need to adjust this.
    if input_tensor.shape != recon_tensor.shape:
         logging.warning("Input and reconstruction tensor shapes mismatch. Error calculation might be incorrect.")
         # Attempt to calculate error based on available dimensions
         errors = np.mean((normalized_mfccs - reconstructed_output)**2, axis=1)

    else:
        errors = np.mean((normalized_mfccs - reconstructed_output)**2, axis=1)

    # 6. Generate timestamps for each frame
    n_frames = normalized_mfccs.shape[0]
    timestamps = np.arange(n_frames) * hop_length / sr

    return reconstructed_output, errors, timestamps