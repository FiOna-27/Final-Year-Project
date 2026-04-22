import os, sys, json, argparse, time, re
import numpy as np
import h5py
import librosa
from pathlib import Path
from tqdm import tqdm

#    Must match phase3_nsynth_model.py and app.py                              
SAMPLE_RATE = 16000
HOP_LENGTH  = 512
FFT_SIZE    = 2048
N_MELS      = 64
FMIN        = 27.5
FMAX        = 4200.0
N_FRAMES    = 32
MIDI_MIN    = 21    # A0
MIDI_MAX    = 108   # C8
N_NOTES     = 88

# Only use keyboard/piano family instruments
PIANO_FAMILIES = {'keyboard'}

# NSynth sample rate is 16000 — no resampling needed
NSYNTH_SR = 16000


#    Audio to mel window                                                         

def wav_to_mel_window(audio_path: str) -> np.ndarray:
    # Load audio, convert to mel spectrogram, normalise to [0,1], and extract a (N_MELS, N_FRAMES) window centred on the note attack.
    try:
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # Amplitude normalisation — matches app.py and build_dataset.py
        y = y / (np.max(np.abs(y)) + 1e-6)

        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE,
            n_fft=FFT_SIZE, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

        T = mel_db.shape[1]

        # NSynth samples are 4 seconds. The note attack is at ~0.05s.
        # Centre the window at frame 3 (0.1s) to capture the onset + early sustain.
        # This is more representative of what the live mic sees.
        attack_frame = max(N_FRAMES, int(0.10 * SAMPLE_RATE / HOP_LENGTH))
        start = max(0, attack_frame - N_FRAMES // 2)
        end   = start + N_FRAMES
        if end > T:
            start = max(0, T - N_FRAMES)
            end   = T

        window = mel_db[:, start:end]
        if window.shape[1] < N_FRAMES:
            window = np.pad(window, ((0,0),(0, N_FRAMES - window.shape[1])))

        return window.astype(np.float32)

    except Exception as e:
        return None


def extract_pitch_from_filename(filename: str):
   # Extract MIDI pitch from filename using regex. NSynth filenames have the format: "instrument-pitch-velocity-uniqueid.wav". We want the pitch number, which is a three-digit integer. If the filename doesn't match or the pitch is out of MIDI range, return None.
    m = re.search(r'-(\d{3})-\d{3}\.wav$', filename)
    if not m:
        return None
    pitch = int(m.group(1))
    if pitch < MIDI_MIN or pitch > MIDI_MAX:
        return None
    return pitch


#    Build HDF5 for one NSynth split                                           

def build_hdf5_nsynth(nsynth_dir: Path, out_path: str, split_label: str, max_files=None):
    # Scan the audio/ folder for WAV files, filter for piano instruments using examples.json if available, and extract mel windows and pitch labels to write to HDF5. Each example is a single note, so the label is a single MIDI pitch class (0-87). The output HDF5 has two resizable datasets: 'windows' of shape (n_samples, N_MELS, N_FRAMES) and 'labels' of shape (n_samples,).
    audio_dir = nsynth_dir / 'audio'
    if not audio_dir.exists():
        print(f'  ⚠  audio/ folder not found in {nsynth_dir}')
        return 0

    # Load examples.json to filter piano instruments only
    piano_files = set()
    json_path = nsynth_dir / 'examples.json'
    if json_path.exists():
        print(f'  Loading examples.json …')
        with open(json_path) as f:
            examples = json.load(f)
        for name, meta in examples.items():
            if meta.get('instrument_family_str') in PIANO_FAMILIES:
                piano_files.add(name + '.wav')
        print(f'  {len(piano_files):,} piano samples found in examples.json')
    else:
        # Fallback: assume all keyboard_* files are piano
        print('  examples.json not found — using filename filter (keyboard_*)')
        piano_files = None   # will filter by name below

    # Collect WAV paths
    all_wavs = sorted(audio_dir.glob('*.wav'))
    if piano_files is not None:
        wavs = [w for w in all_wavs if w.name in piano_files]
    else:
        wavs = [w for w in all_wavs if w.name.startswith('keyboard_')]

    print(f'  [{split_label}] {len(wavs):,} piano WAV files to process')

    if max_files:
        wavs = wavs[:max_files]
        print(f'  Capped at {max_files} files (test mode)')

    os.makedirs(Path(out_path).parent, exist_ok=True)
    total_written = 0
    skipped = 0

    with h5py.File(out_path, 'w') as hf:
        # Resizable datasets — label is a single int per example (pitch class index 0-87)
        ds_X = hf.create_dataset('windows', shape=(0, N_MELS, N_FRAMES),
            maxshape=(None, N_MELS, N_FRAMES), dtype='float32',
            chunks=(256, N_MELS, N_FRAMES), compression='gzip')
        ds_Y = hf.create_dataset('labels', shape=(0,),
            maxshape=(None,), dtype='int64',
            chunks=(256,))
        hf.attrs['n_mels']   = N_MELS
        hf.attrs['n_frames'] = N_FRAMES
        hf.attrs['split']    = split_label
        hf.attrs['task']     = 'monophonic_pitch'   # marks this as NSynth dataset

        for wav_path in tqdm(wavs, desc=f'  {split_label}'):
            pitch = extract_pitch_from_filename(wav_path.name)
            if pitch is None:
                skipped += 1
                continue

            window = wav_to_mel_window(str(wav_path))
            if window is None:
                skipped += 1
                continue

            label = pitch - MIDI_MIN   # convert to 0-87 index

            # Append to HDF5
            ds_X.resize(total_written + 1, axis=0)
            ds_Y.resize(total_written + 1, axis=0)
            ds_X[total_written] = window
            ds_Y[total_written] = label
            total_written += 1

        hf.attrs['n_samples'] = total_written

    if total_written == 0:
        Path(out_path).unlink(missing_ok=True)
        print(f'  ⚠  No valid samples written for {split_label}')
        return 0

    size_mb = Path(out_path).stat().st_size / 1e6
    print(f'  ✅ {total_written:,} samples saved ({size_mb:.1f} MB)'
          f'  [{skipped} skipped]')
    return total_written


#    Main                                                                      

def main():
    parser = argparse.ArgumentParser(description='NSynth Piano to HDF5')

    # Single-dir mode (all splits in one folder, rare)
    parser.add_argument('--nsynth_dir',   default=None,
        help='NSynth directory if all splits are together')

    # Multi-dir mode (separate train/valid/test downloads, common)
    parser.add_argument('--nsynth_train', default=None,
        help='Path to nsynth-train folder')
    parser.add_argument('--nsynth_valid', default=None,
        help='Path to nsynth-valid folder')
    parser.add_argument('--nsynth_test',  default=None,
        help='Path to nsynth-test folder')

    parser.add_argument('--output_dir',   required=True,
        help='Where to write nsynth_train.h5, nsynth_valid.h5, nsynth_test.h5')
    parser.add_argument('--max_files',    type=int, default=None,
        help='Cap files per split (e.g. 5000 for quick test)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('\n' + '═'*60)
    print('  NSynth Piano to HDF5 Dataset Builder')
    print('═'*60)
    print(f'  Config: N_MELS={N_MELS}  N_FRAMES={N_FRAMES}  SR={SAMPLE_RATE}')
    print(f'  Output: {output_dir}')

    t0 = time.time()
    totals = {}

    # Build list of (nsynth_dir, split_label, output_filename)
    splits = []
    if args.nsynth_dir:
        d = Path(args.nsynth_dir)
        splits = [(d/'train', 'train', 'nsynth_train.h5'),
                  (d/'valid', 'valid', 'nsynth_valid.h5'),
                  (d/'test',  'test',  'nsynth_test.h5')]
    else:
        if args.nsynth_train:
            splits.append((Path(args.nsynth_train), 'train', 'nsynth_train.h5'))
        if args.nsynth_valid:
            splits.append((Path(args.nsynth_valid), 'valid', 'nsynth_valid.h5'))
        if args.nsynth_test:
            splits.append((Path(args.nsynth_test),  'test',  'nsynth_test.h5'))

    if not splits:
        print('ERROR: provide --nsynth_dir or at least one of --nsynth_train/valid/test')
        sys.exit(1)

    for nsynth_dir, split_label, out_name in splits:
        if not nsynth_dir.exists():
            print(f'\n  Skipping {split_label} — {nsynth_dir} not found')
            continue
        print(f'\n  Processing {split_label} split from {nsynth_dir}')
        out_path = str(output_dir / out_name)
        n = build_hdf5_nsynth(nsynth_dir, out_path, split_label, args.max_files)
        totals[split_label] = n

    elapsed = time.time() - t0
    print(f'\n{"═"*60}')
    print(f'  Done in {elapsed/60:.1f} min')
    for split, n in totals.items():
        print(f'  {split:8s}: {n:,} samples')
    print('═'*60)
    print("""
  NEXT STEP — train the NSynth model:

    python phase4_1_train.py \\
        --train_h5   "{out}/nsynth_train.h5" \\
        --val_h5     "{out}/nsynth_valid.h5" \\
        --output_dir "{out}/nsynth_checkpoints" \\
        --epochs 30
""".format(out=str(output_dir)))


if __name__ == '__main__':
    main()
