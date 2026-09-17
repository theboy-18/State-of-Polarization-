import pandas as pd
import numpy as np

def save_anomalous_segments_dataset(chunks, sr=44100, filename="anomalies.xlsx"):
    """
    Extract anomalous segments from a list of chunk dictionaries and save
    them as a full dataset in Excel.
    """

    anomaly_segments = []

    for chunk_idx, chunk in enumerate(chunks):
        signal = np.array(chunk['filtered_signal'])
        flags = np.array(chunk['anomaly_flags']).astype(bool)

        inside_anomaly = False
        start_idx = None

        for i, flag in enumerate(flags):
            if flag and not inside_anomaly:
                inside_anomaly = True
                start_idx = i
            elif not flag and inside_anomaly:
                inside_anomaly = False
                end_idx = i - 1

                global_start = chunk_idx * len(signal) + start_idx
                global_end = chunk_idx * len(signal) + end_idx

                anomaly_segments.append({
                    "Chunk_ID": chunk_idx + 1,
                    "Start_Sample": global_start,
                    "End_Sample": global_end,
                    "Start_Time_sec": round(global_start / sr, 6),
                    "End_Time_sec": round(global_end / sr, 6),
                    "Signal_Values": signal[start_idx:end_idx+1].tolist()
                })

        # Handle if anomaly continues to end of chunk
        if inside_anomaly:
            global_start = chunk_idx * len(signal) + start_idx
            global_end = chunk_idx * len(signal) + len(signal) - 1

            anomaly_segments.append({
                "Chunk_ID": chunk_idx + 1,
                "Start_Sample": global_start,
                "End_Sample": global_end,
                "Start_Time_sec": round(global_start / sr, 6),
                "End_Time_sec": round(global_end / sr, 6),
                "Signal_Values": signal[start_idx:].tolist()
            })

    df = pd.DataFrame(anomaly_segments)
    df.to_excel(filename, index=False)
    print(f"Saved {len(df)} anomaly segments to {filename}")