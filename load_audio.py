import librosa
import numpy as np

def stream_stereo_flac(file_path_list):
    """
    Processes a list of stereo FLAC files, computes the difference signal (right - left)
    for each, and returns 1-minute non-overlapping chunks in a list.

    Args:
        file_path_list (list): A list of paths to the stereo FLAC audio files.

    Returns:
        list: A list of 1-minute chunks (NumPy arrays) of the combined difference signal.
    """
    import librosa
    import numpy as np

    sr = 44100 #sampling rate
    minute_samples = sr * 60  # 1 minute in samples
    all_diff_signals = []

    for file_path in file_path_list:
        try:
            signal, _ = librosa.load(file_path, sr=sr, mono=False)

            if signal.ndim != 2 or signal.shape[0] != 2: #check whether the files has a dimension of 2 at the first part because it is stereo(2-channel)
                print(f"Skipping file '{file_path}': Not a stereo audio file.")
                continue

            diff_signal = signal[1] - signal[0] #apparently this helps remove some common noise.
            all_diff_signals.append(diff_signal.reshape(1, -1))  # Shape (1, N) i think when you do the subtraction the shape will be something like (1,) so you'd need to correct that.

        except FileNotFoundError:
            print(f"ERROR: File not found: {file_path}. Skipping.")
        except Exception as e:
            print(f"ERROR while processing '{file_path}': {e}. Skipping.")

    if not all_diff_signals:
        print("No valid audio files were processed.")
        return []

    # Concatenate along axis=1 (since shape is (1, N))
    combined_diff_signal = np.concatenate(all_diff_signals, axis=1).flatten()

    # Split into 1-minute chunks
    chunks = []
    total_samples = len(combined_diff_signal)
    for start in range(0, total_samples, minute_samples):
        end = start + minute_samples
        chunk = combined_diff_signal[start:end]
        if len(chunk) == minute_samples:
            chunks.append(chunk)
            print(f"Chunk of length {len(chunk)} added to list.")

    return chunks