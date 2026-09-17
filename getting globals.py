import os
import numpy as np
from scipy.signal import butter, filtfilt
import librosa
import glob

from load_audio import stream_stereo_flac


# ============================================================
# SETTINGS
# ============================================================

SR = 44100
CUTOFF_FREQ = 300
FILTER_ORDER = 4

N_MFCC = 13
N_MELS = 30

FRAME_LENGTH = 882   # 20 ms at 44.1 kHz
HOP_LENGTH = 441     # 10 ms hop, 50% overlap


# ============================================================
# EXTRACT RAW MFCCs
# ============================================================

def extract_raw_mfccs(audio_chunk):
    """
    Convert one audio chunk into the same raw MFCC representation
    used by the anomaly-detection pipeline.

    No normalization is performed here.

    Returns:
        mfccs: shape (12, number_of_frames)
    """

    # --------------------------------------------------------
    # 1. Design low-pass Butterworth filter
    # --------------------------------------------------------

    nyquist = 0.5 * SR
    normal_cutoff = CUTOFF_FREQ / nyquist

    b, a = butter(
        FILTER_ORDER,
        normal_cutoff,
        btype='low',
        analog=False
    )

    # --------------------------------------------------------
    # 2. Apply zero-phase filtering
    # --------------------------------------------------------

    filtered_signal = filtfilt(
        b,
        a,
        audio_chunk
    )

    # --------------------------------------------------------
    # 3. Calculate MFCCs
    # --------------------------------------------------------

    mfccs = librosa.feature.mfcc(
        y=filtered_signal,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=FRAME_LENGTH,
        n_mels=N_MELS,
        window='hamming',
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------------
    # 4. Remove MFCC 0
    # --------------------------------------------------------

    mfccs = mfccs[1:, :]

    return mfccs


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # PUT YOUR NORMAL TRAINING FILES HERE
    # --------------------------------------------------------

    file_paths = glob.glob("training_audio/*.flac")

    # --------------------------------------------------------
    # Initialize global min/max
    # --------------------------------------------------------

    global_min = np.inf
    global_max = -np.inf

    total_chunks = 0

    # --------------------------------------------------------
    # Process every training chunk
    # --------------------------------------------------------

    for chunk in stream_stereo_flac(file_paths):

        total_chunks += 1

        print(f"\nProcessing chunk {total_chunks}...")

        # Extract raw MFCCs
        mfccs = extract_raw_mfccs(chunk)

        # Find min/max for this chunk
        chunk_min = np.min(mfccs)
        chunk_max = np.max(mfccs)

        print(f"Chunk min: {chunk_min:.6f}")
        print(f"Chunk max: {chunk_max:.6f}")

        # ----------------------------------------------------
        # Update GLOBAL min/max
        # ----------------------------------------------------

        global_min = min(global_min, chunk_min)
        global_max = max(global_max, chunk_max)

    # --------------------------------------------------------
    # Make sure we actually processed something
    # --------------------------------------------------------

    if total_chunks == 0:
        raise RuntimeError(
            "No training chunks were processed. "
            "Check your file paths and training data."
        )

    # --------------------------------------------------------
    # Display final global values
    # --------------------------------------------------------

    print("\n" + "=" * 50)
    print("GLOBAL NORMALIZATION PARAMETERS")
    print("=" * 50)

    print(f"Global min: {global_min:.6f}")
    print(f"Global max: {global_max:.6f}")

    # --------------------------------------------------------
    # Save normalization parameters
    # --------------------------------------------------------

    os.makedirs("training_params", exist_ok=True)

    output_file = "training_params/normalization_params.txt"

    with open(output_file, "w") as f:
        f.write(f"maxi: {global_max}\n")
        f.write(f"mini: {global_min}\n")

    print("\nSaved normalization parameters to:")
    print(output_file)

    print("\nFile contents:")

    with open(output_file, "r") as f:
        print(f.read())
