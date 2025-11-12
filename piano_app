import streamlit as st
import soundfile as sf
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from piano_detector_local import PitchClassifier, extract_features_batch, MODEL_PATH, device

# ------------------------------------
# Streamlit UI setup
# ------------------------------------
st.set_page_config(page_title="🎹 AI Piano Pitch Detector", layout="centered")
st.title("🎶 Piano Pitch Detector (AI Powered)")
st.write("Upload a .wav file, and I’ll tell you what piano note it is 🎧")

uploaded_file = st.file_uploader("Upload your .wav file here", type=["wav"])

# ------------------------------------
# Prediction logic (reuses your model)
# ------------------------------------
def predict_from_audio(file):
    y_audio, sr = sf.read(file)
    if len(y_audio) < 4000:
        st.warning("Audio file too short — try a longer sample.")
        return None

    if y_audio.ndim > 1:
        y_audio = np.mean(y_audio, axis=1)

    feats = extract_features_batch([y_audio], [sr])
    feats = torch.tensor(feats, dtype=torch.float32, device=device)

    # Load model
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
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    note_name, octave = notes[midi % 12], midi // 12 - 1
    return note_name, octave, midi, y_audio, sr

# ------------------------------------
# Main UI workflow
# ------------------------------------
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🎵 Predict Pitch"):
        with st.spinner("Analyzing your piano note... please wait ⏳"):
            try:
                result = predict_from_audio(uploaded_file)
                if result:
                    note, octave, midi, y_audio, sr = result
                    st.success(f"Predicted Note: **{note}{octave}** (MIDI {midi})")

                    # Plot waveform
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.plot(y_audio, color="royalblue")
                    ax.set_title(f"Waveform of Uploaded Audio ({len(y_audio)} samples)")
                    ax.set_xlabel("Samples")
                    ax.set_ylabel("Amplitude")
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Error: {e}")

st.caption("Model loaded from: `piano_pitch_nn.pth`")
