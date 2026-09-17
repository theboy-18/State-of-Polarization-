import numpy as np
import pandas as pd
import logging
from scipy.signal import butter, filtfilt
import librosa
import matplotlib.pyplot as plt
from load_audio import stream_stereo_flac
import time
import os
import glob
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_audio_for_anomaly(audio_chunk, sr=44100, cutoff_freq=300, order=4, n_mfcc=13, n_mels=30, min_val=None, max_val=None):
    """
    Processes a list of differential audio chunks for anomaly detection.

    Args:
        audio_chunks (list): List of 1D NumPy arrays (audio chunks).
        sr (int): Sampling rate of the audio.
        cutoff_freq (int): Cutoff frequency for the low-pass filter.
        order (int): Order of the Butterworth filter.
        n_mfcc (int): Number of MFCC coefficients to extract (including energy term).
        n_mels (int): Number of mel bands to use.
        min_val (float, optional): Global minimum for normalization.
        max_val (float, optional): Global maximum for normalization.

    Returns:
        list: A list of tuples, where each tuple contains:
              (normalized_mfccs, filtered_signal, sr, frame_length, hop_length, used_min, used_max)
    """

    all_processed_data = []
    frame_length = 882  # 20 ms at 44.1 kHz
    hop_length = 441    # 50% overlap
    try:
        logging.info(f"Processing chunk of length {len(audio_chunk)}...")


        if len(audio_chunk) < frame_length:
            logging.warning(f"Skipping short chunk of length {len(audio_chunk)}.")
            return None


        # Low-pass filter
        nyquist = 0.5 * sr
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)#digital butterworth filter
        filtered_signal = filtfilt(b, a, audio_chunk) #applies digital butterworth filter


        # MFCC extraction
        # Get process object
        process = psutil.Process(os.getpid())

        # CPU times before
        cpu_times_before = process.cpu_times()
        start_time = time.time()

        # Your task
        mfccs = librosa.feature.mfcc(
            y=filtered_signal,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=frame_length,
            n_mels=n_mels,
            window='hamming',
            hop_length=hop_length
        )

        # CPU times after
        end_time = time.time()
        cpu_times_after = process.cpu_times()

        # CPU time used
        user_time_used = cpu_times_after.user - cpu_times_before.user
        system_time_used = cpu_times_after.system - cpu_times_before.system
        total_cpu_time = user_time_used + system_time_used

        # Wall time
        wall_time = end_time - start_time

        # Overhead percentage
        overhead_ratio = (total_cpu_time / wall_time) * 100
        cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        normalized_overhead = (total_cpu_time / cores) / wall_time * 100
        total_cpu_percent = (total_cpu_time / (wall_time * cores)) * 100
        print(f"Total CPU % for MFCC extraction: {total_cpu_percent:.2f}%")
        print(f"Wall time for MFCC extraction: {wall_time:.4f} seconds")
        print(f"CPU time (per core) for MFCC extraction: {total_cpu_time/cores:.4f} seconds")
        print(f"Normalized Overhead for MFCC extraction: {normalized_overhead:.2f}%")
        # Drop the first MFCC (energy)
        mfccs = mfccs[1:, :]


        # Normalization
        if min_val is None or max_val is None:
            used_min = np.min(mfccs)
            used_max = np.max(mfccs)
        else:
            used_min = min_val
            used_max = max_val
        if used_max - used_min == 0:
            logging.warning("Min and max are equal; skipping normalization.")
            normalized_mfccs = mfccs
        else:
            normalized_mfccs = (mfccs - used_min) / (used_max - used_min)


        # Transpose: shape (n_frames, n_mfcc)
        normalized_mfccs = normalized_mfccs.T
        all_processed_data.append((
            normalized_mfccs,
            filtered_signal,
            sr,
            frame_length,
            hop_length,
            used_min,
            used_max
        ))
        logging.info(f"Chunk processed. MFCC shape: {normalized_mfccs.shape}")
    except Exception as e:
        logging.error(f"Error processing chunk: {e}")
        return None

    return all_processed_data




"""def plot_processed_audio(processed_data_list, file_names=None, save_dir="./plots"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    if file_names is None:
        file_names = [f"File_{i+1}" for i in range(len(processed_data_list))]

    for i, processed_item in enumerate(processed_data_list):
        if processed_item is None:
            continue

        normalized_mfccs, filtered_signal, sr, frame_length, hop_length, used_min, used_max = processed_item

        time_waveform = np.linspace(0, len(filtered_signal) / sr, num=len(filtered_signal))
        mfcc_time = np.arange(normalized_mfccs.shape[0]) * hop_length / sr
        mfcc_freq = np.arange(1, normalized_mfccs.shape[1] + 1)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        fig.suptitle(file_names[i], fontsize=14, fontweight="bold")

        axes[0].plot(time_waveform, filtered_signal, color="blue")
        axes[0].set_ylim(-2, 2)  # Force y-axis between -2 and 2
        axes[0].set_title("Filtered Signal Waveform")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")

        im = axes[1].imshow(
            normalized_mfccs.T,
            aspect='auto',
            origin='lower',
            extent=[mfcc_time[0], mfcc_time[-1], mfcc_freq[0], mfcc_freq[-1]],
            cmap='viridis'
        )
        axes[1].set_title("MFCCs")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("MFCC Coefficient")
        fig.colorbar(im, ax=axes[1], orientation='vertical', label='Normalised Value')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # --- Before saving ---
        save_path = os.path.join(save_dir, f"{file_names[i]}.png")

# Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    print(f"Plots saved in: {os.path.abspath(save_dir)}")"""

"""start_time = time.time()
file_paths = glob.glob("training_audio/*.flac")

print("\nStarting the audio streaming process...")
all_processed_data = []
chunks = stream_stereo_flac(file_paths)
for chunk in chunks:
    processed_data = process_audio_for_anomaly(chunk, min_val= -5.844, max_val=75.318)
    if processed_data:
        all_processed_data.extend(processed_data)
end_time = time.time()  # Stop the timer
elapsed_time = end_time - start_time
print(f"Total processing time: {elapsed_time:.5f} seconds")


def save_mfcc_to_csv(all_processed_data, output_file="mfcc_features2.csv"):

    all_mfccs = []

    for processed_item in all_processed_data:
        if processed_item is None:
            continue

        normalized_mfccs = processed_item[0]

        all_mfccs.append(normalized_mfccs)

    if not all_mfccs:
        print("No MFCC data to save.")
        return

    # Combine all files/chunks vertically
    all_mfccs = np.vstack(all_mfccs)

    # Create column names
    columns = [f"MFCC_{i+1}" for i in range(all_mfccs.shape[1])]

    df = pd.DataFrame(all_mfccs, columns=columns)

    df.to_csv(output_file, index=False)

    print(f"Saved MFCC features to: {output_file}")
    print(f"Final dataset shape: {df.shape}")

if all_processed_data:
    save_mfcc_to_csv(
        all_processed_data, output_file = "mfcc_features2.csv"
    )
    #plot_processed_audio(all_processed_data, file_names=file_paths)"""