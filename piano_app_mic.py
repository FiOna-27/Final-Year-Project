import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import matplotlib.pyplot as plt
from piano_detector_local import PitchClassifier, extract_features_batch, MODEL_PATH, device

st.set_page_config(page_title="🎙 Real-Time Piano Pitch Detector", layout="centered")
st.title("🎵 Mic-Based Piano Note Detector")
st.write("Click record, play a note on your piano, and I’ll guess what it is 👂")

SAMPLE_RATE = 16000
DURATION = st.slider("Recording length (seconds)", 1, 5, 3)

def predict_from_audio(y_audio, sr):
    if len(y_audio) < 4000:
        st.warning("Audio too short!")
        return None
    if y_audio.ndim > 1:
        y_audio = np.mean(y_audio, axis=1)

    feats = extract_features_batch([y_audio], [sr])
    feats = torch.tensor(feats, dtype=torch.float32, device=device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    num_classes = checkpoint["num_classes"]
    label_to_index = checkpoint["label_to_index"]
    index_to_label = {v: k for k, v in label_to_index.items()}

    model = PitchClassifier(feats.shape[1], num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        output = model(feats)
        pred_index = torch.argmax(output, dim=1).item()

    pred_label = index_to_label[pred_index]
    midi = pred_label + 21
    notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    note_name, octave = notes[midi % 12], midi // 12 - 1
    return note_name, octave, midi, y_audio

if st.button("🎙 Record Now"):
    st.info("Recording... play your note!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    st.success("✅ Recording complete!")

    # flatten array
    y_audio = np.squeeze(audio)
    sf.write("temp_recording.wav", y_audio, SAMPLE_RATE)
    st.audio("temp_recording.wav", format="audio/wav")

    with st.spinner("Analyzing..."):
        try:
            note, octave, midi, y_audio = predict_from_audio(y_audio, SAMPLE_RATE)
            st.success(f"Detected Note: **{note}{octave}**  (MIDI {midi})")

            fig, ax = plt.subplots(figsize=(6,2))
            ax.plot(y_audio, color="orange")
            ax.set_title("Waveform of your recording")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.caption("Mic input sampled at 16 kHz to match model training.")
