import numpy as np
from load_audio import stream_stereo_flac
from preprocessing_audio import process_audio_for_anomaly
from audio_inference import infer_anomaly_autoencoder
from gmm import get_threshold
from collections import deque
import time


def process_all_chunks(file_paths, mini, maxi, _model, history_size=8000, update_interval=2000):
    all_chunks = []
    step = 0
    history_errors = deque(maxlen=history_size)
    batch_errors = []  # store errors since last threshold update
    threshold = None   # initial threshold

    for chunk in stream_stereo_flac(file_paths):
        result = process_audio_for_anomaly(chunk, min_val=mini, max_val=maxi)
        if result is None or result[0][1] is None or len(result[0][1]) == 0:
            continue

        normalized_mfccs, filtered_signal, *_ = result[0]
        # Check inference time
        start_time = time.time()
        _, errors, _ = infer_anomaly_autoencoder(
            normalized_mfccs, _model, 44100, hop_length=441
        )
        end_time = time.time()
        #print(f"Inference time for chunk of length({len(chunk)}): {end_time - start_time:.5f} seconds")

        hop_length = 441
        signal_length = len(filtered_signal)
        stretched_flags = np.zeros(signal_length, dtype=bool)

        for e in errors:
            history_errors.append(e)
            batch_errors.append(e)

            # Only update threshold when enough new errors have been collected
            if len(batch_errors) >= update_interval or threshold is None:
                threshold = get_threshold(np.array(history_errors), step, q=0.65)
                #print(f"Step: {step}, Threshold: {threshold}")
                batch_errors.clear()
                step += 1

        anomaly_flags = errors > threshold

        # Stretch anomaly flags to match filtered signal length
        for i, flag in enumerate(anomaly_flags):
            start = i * hop_length
            end = min(start + hop_length, signal_length)
            if flag:
                stretched_flags[start:end] = True

        all_chunks.append({
            'filtered_signal': filtered_signal,
            'anomaly_flags': stretched_flags,
            'errors': errors,
            'threshold': threshold
        })

    return all_chunks