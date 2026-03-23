#!/usr/bin/env python3
"""
MAESTRO Piano Note Transcription — Full Training Pipeline
==========================================================
Train a CNN on the MAESTRO dataset to detect piano notes from
mel spectrograms, then export to TensorFlow.js for Piano Coach.

SETUP:
  pip install tensorflow librosa pretty_midi numpy tqdm tensorflowjs

DOWNLOAD MAESTRO:
  https://magenta.tensorflow.org/datasets/maestro
  → maestro-v3.0.0.zip  (~130 GB audio)  or  maestro-v3.0.0-midi.zip (MIDI only for testing)

USAGE:
  python train_maestro.py --maestro_dir ./maestro --output_dir ./output
  python train_maestro.py --maestro_dir ./maestro --output_dir ./output --dry_run
"""

import os, sys, argparse, json, time
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────

class Config:
    # Audio
    SAMPLE_RATE   = 16000
    HOP_LENGTH    = 512         # ~32ms hop at 16kHz
    FFT_SIZE      = 2048
    N_MELS        = 64
    FMIN          = 27.5        # A0 — lowest piano note
    FMAX          = 4200.0      # well above C8

    # CNN input
    N_FRAMES      = 11          # temporal context: ~11 × 32ms ≈ 352ms
    INPUT_SHAPE   = (N_MELS, N_FRAMES, 1)

    # Piano
    MIDI_MIN      = 21          # A0
    MIDI_MAX      = 108         # C8
    N_NOTES       = 88

    # Training
    BATCH_SIZE    = 64
    EPOCHS        = 25
    LR            = 1e-3
    VAL_SPLIT     = 0.1
    MAX_FILES     = None        # set to e.g. 50 for a quick test run

    # Export
    KERAS_PATH    = "model.keras"
    TFJS_DIR      = "web_model"

CFG = Config()

# ─── IMPORTS (deferred to avoid slow startup if just checking args) ────────────

def import_libs():
    global tf, librosa, pretty_midi
    import tensorflow as tf
    import librosa
    import pretty_midi
    print(f"  TensorFlow {tf.__version__} · librosa {librosa.__version__} · GPU: {bool(tf.config.list_physical_devices('GPU'))}")


# ─── AUDIO → MEL SPECTROGRAM ──────────────────────────────────────────────────

def audio_to_mel(audio_path: str) -> np.ndarray:
    """
    Load audio and compute log-mel spectrogram.

    Returns:
        mel  shape: (N_MELS, T)  — T = number of time frames
    """
    y, _ = librosa.load(audio_path, sr=CFG.SAMPLE_RATE, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=CFG.SAMPLE_RATE,
        n_fft=CFG.FFT_SIZE,
        hop_length=CFG.HOP_LENGTH,
        n_mels=CFG.N_MELS,
        fmin=CFG.FMIN,
        fmax=CFG.FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # shape: (64, T)

    # Normalize to [0, 1] per file
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db


# ─── MIDI → PIANO ROLL (88 × T binary matrix) ────────────────────────────────

def midi_to_piano_roll(midi_path: str, n_frames: int) -> np.ndarray:
    """
    Convert MIDI to a binary piano roll aligned to mel spectrogram frames.

    Returns:
        roll  shape: (88, T)  — 1 if note is active at that frame, else 0
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    fs   = CFG.SAMPLE_RATE / CFG.HOP_LENGTH   # frames per second
    roll = midi.get_piano_roll(fs=fs)           # shape: (128, T_midi)

    # Crop to 88 piano keys
    roll = roll[CFG.MIDI_MIN : CFG.MIDI_MAX + 1]   # (88, T_midi)

    # Align frame count (MIDI and audio may differ slightly)
    T = min(roll.shape[1], n_frames)
    out = np.zeros((CFG.N_NOTES, n_frames), dtype=np.float32)
    out[:, :T] = (roll[:, :T] > 0).astype(np.float32)
    return out


# ─── SLICE INTO TRAINING EXAMPLES ────────────────────────────────────────────

def extract_examples(mel: np.ndarray, roll: np.ndarray):
    """
    Slide a window of N_FRAMES over the spectrogram.

    X:  (N_samples, N_MELS, N_FRAMES)  mel window
    Y:  (N_samples, 88)                binary note vector at center frame
    """
    half  = CFG.N_FRAMES // 2
    T     = mel.shape[1]
    X, Y  = [], []

    for t in range(half, T - half):
        window = mel[:, t - half : t + half + 1]   # (64, 11)
        labels = roll[:, t]                          # (88,)
        X.append(window)
        Y.append(labels)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ─── PROCESS ONE FILE PAIR (used for multiprocessing) ────────────────────────

def process_file_pair(args):
    audio_path, midi_path = args
    try:
        mel  = audio_to_mel(audio_path)
        roll = midi_to_piano_roll(midi_path, mel.shape[1])
        X, Y = extract_examples(mel, roll)
        return X, Y
    except Exception as e:
        print(f"\n  ⚠  Skipped {Path(audio_path).name}: {e}", flush=True)
        return None, None


# ─── BUILD DATASET FROM MAESTRO ───────────────────────────────────────────────

def build_dataset(maestro_dir: str):
    """
    Scan MAESTRO directory, load file pairs, return (X_train, Y_train).
    Uses multiprocessing for speed.
    """
    maestro_dir = Path(maestro_dir)
    meta_path   = maestro_dir / "maestro-v3.0.0.json"

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        # MAESTRO v3 JSON is columnar: each key is a field name,
        # each value is a dict of {index: value} — NOT a list of row dicts.
        # e.g. meta["split"]["0"] == "train", meta["audio_filename"]["0"] == "2004/..."
        if "split" in meta and isinstance(meta["split"], dict):
            indices = meta["split"].keys()
            pairs = [
                (str(maestro_dir / meta["audio_filename"][i]),
                 str(maestro_dir / meta["midi_filename"][i]))
                for i in indices
                if meta["split"][i] == "train"
            ]
        else:
            # Older format: list of row dicts
            pairs = [
                (str(maestro_dir / e["audio_filename"]),
                 str(maestro_dir / e["midi_filename"]))
                for e in meta.values()
                if isinstance(e, dict) and e.get("split") == "train"
            ]

        print(f"  Found {len(pairs)} training pairs in JSON metadata")
    else:
        # Fallback: scan directory for matching audio/midi pairs
        audio_files = sorted(maestro_dir.rglob("*.wav")) + sorted(maestro_dir.rglob("*.flac"))
        pairs = []
        for af in audio_files:
            mf = af.with_suffix(".midi")
            if not mf.exists():
                mf = af.with_suffix(".mid")
            if mf.exists():
                pairs.append((str(af), str(mf)))

    if CFG.MAX_FILES:
        pairs = pairs[:CFG.MAX_FILES]

    print(f"  Processing {len(pairs)} file pairs …")
    workers = max(1, min(cpu_count() - 1, 8))
    all_X, all_Y = [], []

    with Pool(workers) as pool:
        for X, Y in tqdm(pool.imap(process_file_pair, pairs), total=len(pairs)):
            if X is not None:
                all_X.append(X)
                all_Y.append(Y)

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    # Add channel dimension for Conv2D
    X = X[..., np.newaxis]  # (N, 64, 11, 1)

    print(f"  Dataset: X {X.shape}  Y {Y.shape}  ({X.nbytes/1e9:.2f} GB)")
    return X, Y


# ─── CNN MODEL (matches Piano Coach browser architecture) ─────────────────────

def build_model():
    import tensorflow as tf

    inputs = tf.keras.Input(shape=CFG.INPUT_SHAPE)                        # (64, 11, 1)

    x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)                             # → (32, 5, 32)

    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)                             # → (16, 2, 64)

    x = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(CFG.N_NOTES, activation='sigmoid')(x)  # sigmoid for multi-note

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CFG.LR),
        loss='binary_crossentropy',         # multi-label: each note independent
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model


# ─── TRAINING ─────────────────────────────────────────────────────────────────

def train(X: np.ndarray, Y: np.ndarray, output_dir: str):
    import tensorflow as tf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model()
    model.summary()
    print(f"\n  Parameters: {model.count_params():,}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / CFG.KERAS_PATH),
            save_best_only=True,
            monitor='val_loss',
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=4, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_dir / 'logs'),
        ),
    ]

    print(f"\n  Training on {len(X):,} examples "
          f"(val {int(len(X)*CFG.VAL_SPLIT):,}) · batch {CFG.BATCH_SIZE} · {CFG.EPOCHS} epochs\n")

    history = model.fit(
        X, Y,
        batch_size=CFG.BATCH_SIZE,
        epochs=CFG.EPOCHS,
        validation_split=CFG.VAL_SPLIT,
        callbacks=callbacks,
        shuffle=True,
    )

    return model, history


# ─── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate(model, X_val, Y_val):
    """Frame-level F1 score (standard for piano transcription)."""
    pred = model.predict(X_val, batch_size=256)
    pred_bin = (pred > 0.5).astype(np.float32)

    tp = (pred_bin * Y_val).sum()
    fp = (pred_bin * (1 - Y_val)).sum()
    fn = ((1 - pred_bin) * Y_val).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\n  Frame-level metrics:")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


# ─── EXPORT TO TENSORFLOW.JS ──────────────────────────────────────────────────

def export_tfjs(model, output_dir: str):
    """
    Export the trained model for use in the Piano Coach browser app.
    Then in piano-coach-v2.html, replace buildModel() with:

        model = await tf.loadLayersModel('web_model/model.json');

    You can serve the web_model/ directory with any static file server:
        python -m http.server 8080
    """
    try:
        import tensorflowjs as tfjs
        tfjs_dir = str(Path(output_dir) / CFG.TFJS_DIR)
        tfjs.converters.save_keras_model(model, tfjs_dir)
        print(f"\n  ✅ TensorFlow.js model saved to: {tfjs_dir}/")
        print(f"     Files: model.json + weight shards")
        print(f"\n  To use in Piano Coach HTML — swap buildModel() with:")
        print(f"     model = await tf.loadLayersModel('web_model/model.json');")
    except ImportError:
        print("\n  ⚠  tensorflowjs not installed.")
        print("     Run: pip install tensorflowjs")
        print("     Then: tensorflowjs_converter --input_format keras "
              f"output/{CFG.KERAS_PATH} output/{CFG.TFJS_DIR}/")


# ─── DRY RUN (no MAESTRO needed — tests pipeline shapes) ──────────────────────

def dry_run():
    """
    Generate synthetic data to verify the full pipeline end-to-end
    without downloading MAESTRO. Useful for smoke testing.
    """
    print("\n  DRY RUN — generating synthetic data …")
    N = 2000
    X = np.random.rand(N, CFG.N_MELS, CFG.N_FRAMES, 1).astype(np.float32)
    # Simulate sparse labels (most frames have 1–3 notes active)
    Y = np.zeros((N, CFG.N_NOTES), dtype=np.float32)
    for i in range(N):
        n_active = np.random.choice([0, 1, 2, 3], p=[0.3, 0.5, 0.15, 0.05])
        if n_active:
            notes = np.random.choice(CFG.N_NOTES, n_active, replace=False)
            Y[i, notes] = 1.0
    print(f"  Synthetic X: {X.shape}  Y: {Y.shape}")
    print(f"  Avg active notes per frame: {Y.sum(1).mean():.2f}")
    return X, Y


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MAESTRO Piano Transcription Training")
    parser.add_argument("--maestro_dir", default="./maestro",
                        help="Path to MAESTRO dataset root directory")
    parser.add_argument("--output_dir",  default="./output",
                        help="Where to save model, logs, TF.js export")
    parser.add_argument("--max_files",   type=int, default=None,
                        help="Limit file pairs (e.g. 50 for a quick test)")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Use synthetic data — no MAESTRO needed")
    parser.add_argument("--epochs",      type=int, default=CFG.EPOCHS)
    parser.add_argument("--batch_size",  type=int, default=CFG.BATCH_SIZE)
    args = parser.parse_args()

    CFG.EPOCHS     = args.epochs
    CFG.BATCH_SIZE = args.batch_size
    if args.max_files:
        CFG.MAX_FILES = args.max_files

    print("\n" + "═"*60)
    print("  MAESTRO Piano Transcription · Training Pipeline")
    print("═"*60)
    t0 = time.time()

    print("\n[1/5] Importing libraries …")
    import_libs()

    # ── Data ──
    print("\n[2/5] Building dataset …")
    if args.dry_run:
        X, Y = dry_run()
    else:
        if not Path(args.maestro_dir).exists():
            print(f"\n  ERROR: maestro_dir not found: {args.maestro_dir}")
            print("  Download MAESTRO from: https://magenta.tensorflow.org/datasets/maestro")
            print("  Or run with --dry_run to test the pipeline without data.\n")
            sys.exit(1)
        X, Y = build_dataset(args.maestro_dir)

    # ── Train ──
    print("\n[3/5] Training model …")
    model, history = train(X, Y, args.output_dir)

    # ── Evaluate ──
    print("\n[4/5] Evaluating …")
    split = int(len(X) * CFG.VAL_SPLIT)
    metrics = evaluate(model, X[:split], Y[:split])

    # Save metrics
    metrics_path = Path(args.output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({**metrics, "epochs": CFG.EPOCHS, "n_train": len(X)}, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # ── Export ──
    print("\n[5/5] Exporting to TensorFlow.js …")
    export_tfjs(model, args.output_dir)

    elapsed = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"  Done in {elapsed/60:.1f} min · F1: {metrics['f1']:.4f}")
    print(f"  Model  : {args.output_dir}/{CFG.KERAS_PATH}")
    print(f"  TF.js  : {args.output_dir}/{CFG.TFJS_DIR}/model.json")
    print("═"*60 + "\n")

    print("  NEXT STEPS:")
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │  1. Serve the web_model/ directory:                  │")
    print("  │       python -m http.server 8080                     │")
    print("  │                                                      │")
    print("  │  2. In piano-coach-v2.html, find buildModel() and    │")
    print("  │     replace the tf.sequential() block with:          │")
    print("  │       model = await tf.loadLayersModel(              │")
    print("  │         'http://localhost:8080/model.json');          │")
    print("  │                                                      │")
    print("  │  3. Open piano-coach-v2.html in your browser         │")
    print("  │     CNN will now use REAL trained MAESTRO weights!   │")
    print("  └──────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
