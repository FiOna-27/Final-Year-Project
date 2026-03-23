#!/usr/bin/env python3
"""
serve_model.py — Piano Coach Inference Server
===============================================
Loads best_model.pt and serves real-time note predictions to the
Piano Coach HTML via a local REST API.

SETUP:
    pip install flask flask-cors torch librosa numpy

USAGE:
    python serve_model.py --model "E:\\TWOFYP\\checkpoints\\best_model.pt"

    Then open piano-coach-v2.html in your browser.
    The HTML will POST audio to http://localhost:5000/predict automatically.

ENDPOINTS:
    GET  /status          — health check, returns model info
    POST /predict         — accepts raw float32 audio, returns note prediction
"""

import argparse
import sys
import numpy as np
from pathlib import Path

import torch
import librosa
from flask import Flask, request, jsonify
from flask_cors import CORS

# phase3_model.py must be in the same folder as this script
sys.path.insert(0, str(Path(__file__).parent))
from phase3_model import PianoTranscriptionCNN

# ─── CONFIG (must match build_dataset.py exactly) ─────────────────────────────

SAMPLE_RATE = 16000
HOP_LENGTH  = 512
FFT_SIZE    = 2048
N_MELS      = 64
FMIN        = 27.5
FMAX        = 4200.0
N_FRAMES    = 11
MIDI_MIN    = 21   # A0
N_NOTES     = 88

NOTE_NAMES  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# ─── GLOBALS ──────────────────────────────────────────────────────────────────

app    = Flask(__name__)
CORS(app)   # allow browser cross-origin requests

model  = None
device = None
model_info = {}

# ─── MEL SPECTROGRAM (identical pipeline to build_dataset.py) ─────────────────

def audio_to_mel_window(audio_samples: np.ndarray, sr: int) -> np.ndarray:
    """
    Convert a chunk of raw audio into the (1, N_MELS, N_FRAMES) tensor
    the model expects.

    The browser sends ~4096 samples (one Web Audio fftSize buffer) at its
    native sample rate. We resample to 16kHz, compute a mel spectrogram,
    and take the centre N_FRAMES frames as a sliding window.
    """
    # Resample if browser sample rate differs from training rate
    if sr != SAMPLE_RATE:
        audio_samples = librosa.resample(audio_samples, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Pad if too short to fill N_FRAMES windows
    min_len = FFT_SIZE + HOP_LENGTH * N_FRAMES
    if len(audio_samples) < min_len:
        audio_samples = np.pad(audio_samples, (0, min_len - len(audio_samples)))

    mel = librosa.feature.melspectrogram(
        y=audio_samples,
        sr=SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1] — same as build_dataset.py
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Take N_FRAMES from the centre of whatever we got
    T = mel_db.shape[1]
    half = N_FRAMES // 2
    centre = T // 2
    start  = max(0, centre - half)
    end    = start + N_FRAMES
    if end > T:
        start = T - N_FRAMES
        end   = T

    window = mel_db[:, start:end]   # (N_MELS, N_FRAMES)

    # Add batch + channel dims → (1, 1, N_MELS, N_FRAMES)
    return window[np.newaxis, np.newaxis, :, :].astype(np.float32)


# ─── INFERENCE ────────────────────────────────────────────────────────────────

def run_inference(audio_samples: np.ndarray, sr: int, yin_midi: int):
    """
    Run the trained CNN on a chunk of audio.

    Returns dict with:
        midi        int   — best MIDI note (fused YIN + CNN)
        note        str   — note name e.g. "C#"
        octave      int
        confidence  float — 0–1
        probabilities list[float] — all 88 note probabilities
    """
    window = audio_to_mel_window(audio_samples, sr)   # (1,1,64,11)
    tensor = torch.from_numpy(window).to(device)

    with torch.no_grad():
        probs = model(tensor)   # (1, 88)

    probs_np = probs.cpu().numpy()[0]   # (88,)

    cnn_idx  = int(np.argmax(probs_np))
    cnn_midi = MIDI_MIN + cnn_idx
    cnn_conf = float(probs_np[cnn_idx])

    # Fuse YIN + CNN: if they agree within 1 semitone, trust YIN's pitch
    # precision and boost confidence
    if yin_midi > 0 and abs(yin_midi - cnn_midi) <= 1:
        fused_midi = yin_midi
        fused_conf = min(0.99, cnn_conf + 0.1)
    elif yin_midi > 0:
        # Disagree — weight by CNN confidence; high conf CNN wins, else trust YIN
        fused_midi = cnn_midi if cnn_conf > 0.6 else yin_midi
        fused_conf = cnn_conf
    else:
        fused_midi = cnn_midi
        fused_conf = cnn_conf

    note   = NOTE_NAMES[fused_midi % 12]
    octave = (fused_midi // 12) - 1

    return {
        "midi":          fused_midi,
        "note":          note,
        "octave":        octave,
        "confidence":    round(fused_conf, 4),
        "probabilities": probs_np.tolist(),   # all 88 for the browser heatmap
    }


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    """Health check — the HTML polls this on load."""
    return jsonify({
        "status":  "ok",
        "model":   model_info,
        "device":  str(device),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON body:
    {
        "audio":    [float, ...],   raw PCM samples (Float32Array from browser)
        "sr":       44100,          browser sample rate
        "yin_midi": 69              YIN pitch estimate (0 if no pitch found)
    }

    Returns:
    {
        "midi":          69,
        "note":          "A",
        "octave":        4,
        "confidence":    0.94,
        "probabilities": [0.01, 0.02, ...]   (88 values)
    }
    """
    try:
        data      = request.get_json(force=True)
        audio     = np.array(data["audio"],    dtype=np.float32)
        sr        = int(data.get("sr",        44100))
        yin_midi  = int(data.get("yin_midi",  0))

        result = run_inference(audio, sr, yin_midi)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── STARTUP ──────────────────────────────────────────────────────────────────

def load_model(model_path: str):
    global model, device, model_info

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    n_mels   = ckpt.get("n_mels",   N_MELS)
    n_frames = ckpt.get("n_frames", N_FRAMES)

    model = PianoTranscriptionCNN(n_mels=n_mels, n_frames=n_frames).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    model_info = {
        "path":     model_path,
        "epoch":    ckpt.get("epoch",  "?"),
        "f1":       round(float(ckpt.get("f1", 0)), 4),
        "n_mels":   n_mels,
        "n_frames": n_frames,
        "params":   params,
    }

    print(f"  Model loaded: epoch {model_info['epoch']} · F1={model_info['f1']} · {params:,} params")
    print(f"  Path: {model_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Piano Coach inference server")
    parser.add_argument("--model", required=True,
                        help="Path to best_model.pt from phase4_train.py")
    parser.add_argument("--port",  type=int, default=5000)
    parser.add_argument("--host",  default="127.0.0.1")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"ERROR: model file not found: {args.model}")
        print("Run phase4_train.py first to generate best_model.pt")
        sys.exit(1)

    print("\n" + "═"*55)
    print("  Piano Coach Inference Server")
    print("═"*55)
    load_model(args.model)

    print(f"  Starting on http://{args.host}:{args.port}")
    print(f"  Open piano-coach-v2.html in your browser")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
