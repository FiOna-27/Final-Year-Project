import os, sys, argparse, time
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

#    librosa / pretty_midi imported at TOP so subprocesses inherit them         
import librosa
import pretty_midi

#     CONFIG                                                                    

SAMPLE_RATE  = 16000
HOP_LENGTH   = 512
FFT_SIZE     = 2048
N_MELS       = 64
FMIN         = 27.5    # A0
FMAX         = 4200.0  # well above C8
N_FRAMES     = 32      # ↑ from 11 — more temporal context (~1 second)
MIDI_MIN     = 21      # A0
MIDI_MAX     = 108     # C8
N_NOTES      = 88

#     AUDIO → MEL SPECTROGRAM                                                   

def audio_to_mel(audio_path: str) -> np.ndarray:
    """Load audio and return normalised log-mel spectrogram (N_MELS × T)."""
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    # Amplitude normalisation — matches what app.py does at inference time.
    # Without this, quiet recordings look like silence to the model.
    y = y / (np.max(np.abs(y)) + 1e-6)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_fft=FFT_SIZE, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype(np.float32)   # (N_MELS, T)


#     MIDI → PIANO ROLL                                                         

def midi_to_piano_roll(midi_path: str, n_frames: int) -> np.ndarray:
    """Return binary piano roll (88 × T) aligned to mel frames."""
    midi = pretty_midi.PrettyMIDI(midi_path)
    fs   = SAMPLE_RATE / HOP_LENGTH
    roll = midi.get_piano_roll(fs=fs)             # (128, T_midi)
    roll = roll[MIDI_MIN : MIDI_MAX + 1]          # (88, T_midi)
    T    = min(roll.shape[1], n_frames)
    out  = np.zeros((N_NOTES, n_frames), dtype=np.float32)
    out[:, :T] = (roll[:, :T] > 0).astype(np.float32)
    return out


#     SLIDING WINDOW EXTRACTION                                                 

def extract_examples(mel: np.ndarray, roll: np.ndarray):
    """
    Slide an N_FRAMES window over the spectrogram.

    Returns:
        X  (N_samples, N_MELS, N_FRAMES)
        Y  (N_samples, 88)
    """
    half = N_FRAMES // 2
    T    = mel.shape[1]
    X, Y = [], []
    for t in range(half, T - half):
        window = mel[:, t - half : t - half + N_FRAMES]  # always exactly N_FRAMES wide
        if window.shape[1] != N_FRAMES:
            continue   # skip incomplete windows at edges
        X.append(window)
        Y.append(roll[:, t])
    return (np.array(X, dtype=np.float32),
            np.array(Y, dtype=np.float32))


#     PROCESS ONE FILE PAIR                                                     

def process_pair(audio_path: str, midi_path: str):
    """Returns (X, Y) or (None, None) on error."""
    try:
        mel  = audio_to_mel(audio_path)
        roll = midi_to_piano_roll(midi_path, mel.shape[1])
        return extract_examples(mel, roll)
    except Exception as e:
        name = Path(audio_path).name
        print(f"\n  ⚠  Skipped {name}: {e}", flush=True)
        return None, None


#     LOAD MAESTRO METADATA                                                     

def load_pairs(maestro_dir: Path, split: str, max_files=None):
    """
    Return list of (audio_path, midi_path) for the requested split.
    Handles both columnar-JSON (v3) and row-dict-JSON (v1/v2) formats,
    plus a fallback directory scan if no JSON is found.
    """
    meta_path = maestro_dir / "maestro-v3.0.0.json"
    if not meta_path.exists():
        # Try v2 / v1 name
        for name in ["maestro-v2.0.0.json", "maestro-v1.0.0.json"]:
            if (maestro_dir / name).exists():
                meta_path = maestro_dir / name
                break

    pairs = []

    if meta_path.exists():
        import json
        with open(meta_path) as f:
            meta = json.load(f)

        # Columnar format (v3): meta["split"] is a dict of {idx: value}
        if "split" in meta and isinstance(meta["split"], dict):
            for idx in meta["split"]:
                if meta["split"][idx] == split:
                    pairs.append((
                        str(maestro_dir / meta["audio_filename"][idx]),
                        str(maestro_dir / meta["midi_filename"][idx]),
                    ))
        # Row format (v1/v2): meta is a dict of {idx: row_dict}
        else:
            for entry in meta.values():
                if isinstance(entry, dict) and entry.get("split") == split:
                    pairs.append((
                        str(maestro_dir / entry["audio_filename"]),
                        str(maestro_dir / entry["midi_filename"]),
                    ))

        print(f"  [{split}] {len(pairs)} pairs found in JSON")

    else:
        # Fallback: scan subdirectories
        print("  No metadata JSON found — scanning directory …")
        audio_files = sorted(maestro_dir.rglob("*.wav")) + \
                      sorted(maestro_dir.rglob("*.flac"))
        for af in audio_files:
            mf = af.with_suffix(".midi")
            if not mf.exists():
                mf = af.with_suffix(".mid")
            if mf.exists():
                pairs.append((str(af), str(mf)))
        # No split info available from filesystem — use all
        print(f"  [{split}] {len(pairs)} pairs found via directory scan "
              "(no split info — using all for every split)")

    if max_files:
        pairs = pairs[:max_files]
    return pairs





#     WRITE HDF5                                                                

def build_hdf5(pairs, out_path: str, split_label: str):
    """
    Process file pairs and write to HDF5 one file at a time.
    Uses HDF5 resizable datasets so data is appended directly to disk —
    RAM usage is bounded to one audio file at a time, not the whole dataset.
    """
    os.makedirs(Path(out_path).parent, exist_ok=True)
    print(f"\n  Processing {len(pairs)} {split_label} pairs …")

    total_written = 0

    with h5py.File(out_path, "w") as hf:
        # Create resizable datasets — maxshape=None means unlimited rows
        ds_X = hf.create_dataset(
            "windows",
            shape=(0, N_MELS, N_FRAMES),
            maxshape=(None, N_MELS, N_FRAMES),
            dtype="float32",
            chunks=(256, N_MELS, N_FRAMES),
            compression="gzip",
        )
        ds_Y = hf.create_dataset(
            "labels",
            shape=(0, N_NOTES),
            maxshape=(None, N_NOTES),
            dtype="float32",
            chunks=(256, N_NOTES),
            compression="gzip",
        )
        hf.attrs["n_mels"]   = N_MELS
        hf.attrs["n_frames"] = N_FRAMES
        hf.attrs["split"]    = split_label

        for audio_path, midi_path in tqdm(pairs, desc=f"  {split_label}"):
            X, Y = process_pair(audio_path, midi_path)
            if X is None or len(X) == 0:
                continue

            # Append this file's windows/labels to the HDF5 datasets on disk
            n_new = len(X)
            old_size = total_written

            ds_X.resize(total_written + n_new, axis=0)
            ds_Y.resize(total_written + n_new, axis=0)

            ds_X[old_size : old_size + n_new] = X
            ds_Y[old_size : old_size + n_new] = Y

            total_written += n_new
            # Free memory immediately — don't hold any file data
            del X, Y

        hf.attrs["n_samples"] = total_written

    if total_written == 0:
        print(f"  !!  No valid data for split '{split_label}' — skipping.")
        Path(out_path).unlink(missing_ok=True)
        return 0

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"  ✅ Saved {total_written:,} samples  ({size_mb:.1f} MB)")
    return total_written


#MAIN                                                                      

def main():
    parser = argparse.ArgumentParser(description="MAESTRO → HDF5 dataset builder")
    parser.add_argument("--maestro_dir", required=True,
                        help="Root of the MAESTRO dataset (contains year folders)")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory where HDF5 files will be written")
    parser.add_argument("--max_files",   type=int, default=None,
                        help="Cap per split per dataset (e.g. 100 for a quick test)")
    parser.add_argument("--splits",      nargs="+",
                        default=["train", "validation", "test"],
                        help="Which splits to build (default: all three)")
    args = parser.parse_args()

    maestro_dir = Path(args.maestro_dir)
    output_dir  = Path(args.output_dir)

    if not maestro_dir.exists():
        print(f"ERROR: maestro_dir not found: {maestro_dir}")
        sys.exit(1)

    print("\n" + "═"*60)
    print("  MAESTRO → HDF5 Dataset Builder")
    print("═"*60)
    print(f"  librosa {librosa.__version__} · pretty_midi {pretty_midi.__version__}")
    print(f"  Config: N_MELS={N_MELS}  N_FRAMES={N_FRAMES}  SR={SAMPLE_RATE}")
    print(f"  MAESTRO: {maestro_dir}")
    print(f"  Output:  {output_dir}")
    if args.max_files:
        print(f"  Max files per split per dataset: {args.max_files}  (test mode)")

    t0 = time.time()
    totals = {}

    for split in args.splits:
        pairs = load_pairs(maestro_dir, split, args.max_files)

        if not pairs:
            print(f"\n  Skipping '{split}' — no pairs found.")
            continue

        out_path = str(output_dir / f"dataset_{split}.h5")
        n = build_hdf5(pairs, out_path, split)
        totals[split] = n

    elapsed = time.time() - t0
    print(f"\n{'═'*60}")
    print(f"  Done in {elapsed/60:.1f} min")
    for split, n in totals.items():
        print(f"  {split:12s}: {n:,} samples  →  dataset_{split}.h5")
    print("═"*60)

    print("""
  NEXT STEP — run phase4_train.py:

    python phase4_train.py \\
        --train_h5  "{out}/dataset_train.h5" \\
        --val_h5    "{out}/dataset_validation.h5" \\
        --output_dir "{out}/checkpoints" \\
        --epochs 50 --restart
""".format(out=args.output_dir))


if __name__ == "__main__":
    main()
