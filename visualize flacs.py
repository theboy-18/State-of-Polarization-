import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import os

print("Python is running from:")
print(os.getcwd())

print("\nFiles Python can see:")
print(os.listdir())

# ============================================================
# SETTINGS
# ============================================================

SR = 44100

file_paths = [
   "output-20230522-083354.flac",
    "output-20230522-084300.flac",
    "output-20230522-084620.flac"
]


# ============================================================
# LOAD AND VISUALIZE EACH FILE
# ============================================================

for file_path in file_paths:

    print(f"\nLoading: {file_path}")

    # Load stereo audio
    signal, sr = librosa.load(
        file_path,
        sr=SR,
        mono=False
    )

    # Check that the file is actually stereo
    if signal.ndim != 2 or signal.shape[0] != 2:
        print(f"Skipping {file_path}: file is not stereo.")
        continue

    # --------------------------------------------------------
    # Separate channels
    # --------------------------------------------------------

    left = signal[0]
    right = signal[1]

    # Differential signal: Right - Left
    diff_signal = right - left

    # --------------------------------------------------------
    # Time axes
    # --------------------------------------------------------

    time = np.arange(len(left)) / sr

    # --------------------------------------------------------
    # Print some information
    # --------------------------------------------------------

    duration = len(left) / sr

    print(f"Sample rate: {sr} Hz")
    print(f"Number of samples: {len(left):,}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Left min/max: {left.min():.4f} / {left.max():.4f}")
    print(f"Right min/max: {right.min():.4f} / {right.max():.4f}")
    print(f"Difference min/max: {diff_signal.min():.4f} / {diff_signal.max():.4f}")

    # ========================================================
    # FULL SIGNAL
    # ========================================================

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    fig.suptitle(
        f"{os.path.basename(file_path)} - Full Signal",
        fontsize=16
    )

    # Left
    axes[0].plot(time, left, linewidth=0.5)
    axes[0].set_title("Left Channel")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Right
    axes[1].plot(time, right, linewidth=0.5)
    axes[1].set_title("Right Channel")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    # Difference
    axes[2].plot(time, diff_signal, linewidth=0.5)
    axes[2].set_title("Differential Signal: Right - Left")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True)

    plt.tight_layout()

    # Save the figure
    output_name = os.path.splitext(
        os.path.basename(file_path)
    )[0] + "_full.png"

    plt.savefig(
        output_name,
        dpi=200
    )

    plt.show()

    # ========================================================
    # FIRST 1 SECOND - ZOOMED IN
    # ========================================================

    seconds_to_show = 1

    samples_to_show = min(
        int(seconds_to_show * sr),
        len(left)
    )

    zoom_time = time[:samples_to_show]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10))

    fig.suptitle(
        f"{os.path.basename(file_path)} - First {seconds_to_show} Second",
        fontsize=16
    )

    # Left
    axes[0].plot(
        zoom_time,
        left[:samples_to_show],
        linewidth=0.7
    )
    axes[0].set_title("Left Channel")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Right
    axes[1].plot(
        zoom_time,
        right[:samples_to_show],
        linewidth=0.7
    )
    axes[1].set_title("Right Channel")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    # Difference
    axes[2].plot(
        zoom_time,
        diff_signal[:samples_to_show],
        linewidth=0.7
    )
    axes[2].set_title("Differential Signal: Right - Left")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True)

    plt.tight_layout()

    output_name = os.path.splitext(
        os.path.basename(file_path)
    )[0] + "_first_second.png"

    plt.savefig(
        output_name,
        dpi=200
    )

    plt.show()
