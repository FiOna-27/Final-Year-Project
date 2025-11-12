"""
piano_detector_local.py
Detects piano note pitch using a locally stored NSynth dataset.

Usage:
  1. Place nsynth-train.jsonwav.tar.gz in the same folder.
  2. Install dependencies:
     pip install librosa soundfile scikit-learn numpy joblib tqdm matplotlib
  3. Run training:
     python piano_detector_local.py --train --max-samples 2000
  4. Predict:
     python piano_detector_local.py --predict path/to/file.wav
"""

import argparse, os, tarfile, json
import numpy as np
import librosa, soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm

MODEL_PATH = "piano_pitch_rf.joblib"
DATA_ARCHIVE = "nsynth-train.jsonwav.tar.gz"
SAMPLE_RATE = 16000
N_MELS = 128

def extract_features(y, sr=SAMPLE_RATE, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024,
                                         hop_length=256, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)])

def load_nsynth_piano(max_samples=2000):
    """Extract piano samples + metadata from local tar.gz"""
    X, y = [], []
    with tarfile.open(DATA_ARCHIVE, "r:gz") as tar:
        # Find and open JSON file inside the archive
        json_member = [m for m in tar.getmembers() if m.name.endswith(".json")][0]
        with tar.extractfile(json_member) as f:
            metadata = json.load(f)

        wav_members = {m.name.split("/")[-1]: m for m in tar.getmembers() if m.name.endswith(".wav")}
        pianos = [
            k for k, v in metadata.items()
            if any(
                kw in v.get("instrument_family_str", "").lower()
                for kw in ("keyboard", "piano")
            )
        ]

        print(f"Found {len(pianos)} piano samples.")

        for i, key in enumerate(tqdm(pianos[:max_samples], desc="Loading")):
            wav_member = wav_members.get(key + ".wav")
            if not wav_member:
                continue
            with tar.extractfile(wav_member) as wf:
                y_audio, sr = sf.read(wf)
                if y_audio.ndim > 1:
                    y_audio = np.mean(y_audio, axis=1)
                feats = extract_features(y_audio, sr)
                midi_pitch = metadata[key]["pitch"]
                X.append(feats)
                y.append(midi_pitch - 21)  # normalize to 0–87
    return np.stack(X), np.array(y)

def train(max_samples):
    print("🎵 Loading local NSynth dataset…")
    X, y = load_nsynth_piano(max_samples)
    print(f"Collected {len(y)} samples. Training model…")

    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds))
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

def predict(wav_file):
    if not os.path.exists(MODEL_PATH):
        print("⚠️ No model found. Train first.")
        return
    clf = joblib.load(MODEL_PATH)
    y_audio, sr = sf.read(wav_file)
    if y_audio.ndim > 1:
        y_audio = np.mean(y_audio, axis=1)
    if sr != SAMPLE_RATE:
        y_audio = librosa.resample(y_audio.astype(np.float32), sr, SAMPLE_RATE)
    feats = extract_features(y_audio)
    pred = clf.predict(feats.reshape(1, -1))[0]
    midi = pred + 21
    notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    name, octave = notes[midi % 12], midi//12 - 1
    print(f"🎹 Predicted note: {name}{octave} (MIDI {midi})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--predict", type=str)
    p.add_argument("--max-samples", type=int, default=2000)
    args = p.parse_args()

    if args.train:
        train(args.max_samples)
    elif args.predict:
        predict(args.predict)
    else:
        print("Usage: python piano_detector_local.py --train OR --predict file.wav")
