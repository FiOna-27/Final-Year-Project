import streamlit as st
import sounddevice as sd
import numpy as np
import torch
import queue
import threading
import time
import matplotlib.pyplot as plt
from piano_detector_local import PitchClassifier, extract_features_batch, MODEL_PATH, device

st.set_page_config(page_title="🎧 Live Piano Pitch Detector", layout="centered")
st.title("🎹 Real-Time Piano Pitch Tracker")

# Globals
SAMPLE_RATE = 16000
BLOCK_SIZE = 2048       # samples per block (~0.128s)
BUFFER_DURATION = 1.0   # analyze last 1 sec of audio
audio_q = queue.Queue()
live_audio = np.zeros(int(SAMPLE_RATE * BUFFER_DURATION))

# Load model once
checkpoint = torch.load(MODEL_PATH, map_location=device)
num_classes = checkpoint["num_classes"]
label_to_index = checkpoint["label_to_index"]
index_to_label = {v: k for k, v in label_to_index.items()}
# Infer input dimension automatically from checkpoint weights
first_layer_weight = next(iter(checkpoint["model_state_dict"].values()))
input_dim = first_layer_weight.shape[1]  # e.g. 256 from training
model = PitchClassifier(input_dim, num_classes).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

def predict_from_buffer(y_audio, sr):
    feats = extract_features_batch([y_audio], [sr])
    feats = torch.tensor(feats, dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(feats)
        pred_index = torch.argmax(output, dim=1).item()
    pred_label = index_to_label[pred_index]
    midi = pred_label + 21
    notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    note_name, octave = notes[midi % 12], midi // 12 - 1
    return note_name, octave, midi

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata.copy())

def audio_thread(stop_flag):
    global live_audio
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=BLOCK_SIZE, callback=audio_callback):
        while not stop_flag.is_set():
            try:
                block = audio_q.get(timeout=0.1).flatten()
                live_audio = np.roll(live_audio, -len(block))
                live_audio[-len(block):] = block
            except queue.Empty:
                pass

st.write("Press **Start Listening** and play notes — the app will guess live! 🎶")
placeholder_text = st.empty()
placeholder_plot = st.empty()

running = False

if st.button("▶️ Start Listening"):
    stop_flag = threading.Event()
    thread = threading.Thread(target=audio_thread, args=(stop_flag,))
    thread.start()
    st.success("Listening... press Stop to end.")

    running = True

stop_button = st.button("⏹ Stop Listening")

while running:
    if stop_button:
        stop_flag.set()
        st.warning("⏹ Stopped listening.")
        running = False
        break

    if len(live_audio) > 4000:
        note, octave, midi = predict_from_buffer(live_audio, SAMPLE_RATE)
        placeholder_text.markdown(f"### 🎵 Current Note: **{note}{octave}** (MIDI {midi})")

        fig, ax = plt.subplots(figsize=(6,2))
        ax.plot(live_audio, color="violet")
        ax.set_title("Live Audio Waveform (last 1 sec)")
        placeholder_plot.pyplot(fig)

    time.sleep(0.3)
