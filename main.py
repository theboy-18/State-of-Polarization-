import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from load_model import load_model
from process_all_chunks import process_all_chunks
from collections import deque
import time


# Load normalization parameters
@st.cache_data
def load_normalization_params():
    maxi = None
    mini = None
    with open('training_params/normalization_params.txt', 'r') as f:
        for line in f:
            if line.startswith('maxi:'):
                maxi = float(line.split(':')[1].strip())
            elif line.startswith('mini:'):
                mini = float(line.split(':')[1].strip())
    return maxi, mini


# Load model
@st.cache_resource
def load_trained_model():
    return load_model('model.pth')


# File paths
file_paths = ['./test/output-1.flac',
              './test/output-2.flac',
              './test/output-3.flac',
              './test/output-6.flac'
              ]


# Start timing the entire process
start_time = time.time()


# Load data
maxi, mini = load_normalization_params()

# Load model
model = load_trained_model()

# Process all chunks
processed_chunks = process_all_chunks(file_paths, mini, maxi, model)

# End timing after processing
end_time = time.time()  # Stop the timer
elapsed_time = end_time - start_time
print(f"Total processing time: {elapsed_time:.5f} seconds")

# UI state
#Check total time taken for rendering
start_time = time.perf_counter()
if 'current_index' not in st.session_state:

    st.session_state.current_index = 0

#Rendering the page
def show_plot(filtered_signal, anomaly_flags):
    t = np.arange(len(filtered_signal)) / 44100

    # 1. PLOT: Filtered signal only
    st.subheader("Filtered SOP Signal")
    fig1, ax1 = plt.subplots(figsize=(20, 5))
    ax1.plot(t, filtered_signal, color='blue', linewidth=0.7)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Filtered SOP Signal")
    ax1.set_ylim(-2, 2)
    st.pyplot(fig1)

    # 2. PLOT: Anomalies highlighted (red vs blue)
    st.subheader("Filtered SOP After Anomaly Detection")
    fig2, ax2 = plt.subplots(figsize=(20, 5))

    # Split signal into segments based on anomaly flags
    current_start = 0
    current_flag = anomaly_flags[0]

    for i in range(1, len(filtered_signal)):
        if anomaly_flags[i] != current_flag:
            segment = filtered_signal[current_start:i]
            times = t[current_start:i]
            color = 'red' if current_flag else 'blue'
            ax2.plot(times, segment, color=color, linewidth=0.7)
            current_start = i
            current_flag = anomaly_flags[i]

    # Final segment
    segment = filtered_signal[current_start:]
    times = t[current_start:]
    color = 'red' if current_flag else 'blue'
    ax2.plot(times, segment, color=color, linewidth=0.7)
    ax2.set_ylim(-2, 2)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Anomalies Highlighted in SOP Signal")
    st.pyplot(fig2)



# Show UI
if processed_chunks:
    chunk = processed_chunks[st.session_state.current_index]
    show_plot(chunk['filtered_signal'], chunk['anomaly_flags'])

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("Previous", use_container_width=True) and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
            st.rerun()


    with col3:
        if st.button("Next", use_container_width=True) and st.session_state.current_index < len(processed_chunks) - 1:
            st.session_state.current_index += 1
            st.rerun()

else:
    st.warning("No audio chunks found or processed.")

# End timing after rendering the page
end_time = time.perf_counter()
render_time = end_time - start_time
st.write(f"⏱ Render time: {render_time:.5f} seconds")