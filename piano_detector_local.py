import argparse, os, tarfile, json
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

MODEL_PATH = "piano_pitch_nn.pth"
DATA_ARCHIVE = "nsynth-train.jsonwav.tar.gz"
EXTRACTED_FOLDER = "nsynth_train_wavs"
FEATURE_FOLDER = "features"
SAMPLE_RATE = 16000
N_MELS = 128
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Neural Network Model
# -------------------------------
class PitchClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(PitchClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------
# Step 1: Unpack NSynth dataset once
# -------------------------------
def extract_dataset_once():
    if not os.path.exists(EXTRACTED_FOLDER):
        print("Extracting dataset to disk…")
        if not os.path.exists(DATA_ARCHIVE):
            raise FileNotFoundError(f"Dataset archive {DATA_ARCHIVE} not found in {os.getcwd()}")
        try:
            with tarfile.open(DATA_ARCHIVE, "r:gz") as tar:
                tar.extractall(EXTRACTED_FOLDER)
            print(f"Extracted to {EXTRACTED_FOLDER}")
            # Verify JSON file exists
            json_path = os.path.join(EXTRACTED_FOLDER, "nsynth-train", "examples.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"Expected file {json_path} not found after extraction")
        except tarfile.TarError as e:
            raise RuntimeError(f"Failed to extract {DATA_ARCHIVE}: {e}")
    else:
        print(f"Dataset already extracted in {EXTRACTED_FOLDER}")

# -------------------------------
# Step 2: GPU Batch Feature Extraction
# -------------------------------
def extract_features_batch(audio_list, sr_list):
    tensors = []
    for y, sr in zip(audio_list, sr_list):
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        if sr != SAMPLE_RATE:
            y_tensor = torchaudio.functional.resample(y_tensor, sr, SAMPLE_RATE)
        if y_tensor.ndim > 1:
            y_tensor = torch.mean(y_tensor, dim=0)
        tensors.append(y_tensor)

    batch = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=N_MELS
    ).to(device)
    mels = mel_transform(batch)
    log_mels = torch.log(mels + 1e-9)
    feats = torch.cat([log_mels.mean(dim=-1), log_mels.std(dim=-1)], dim=1)
    return feats.cpu().numpy()  # Return as numpy for precomputation

# -------------------------------
# Step 3: Precompute Features
# -------------------------------
def precompute_features(max_samples=2000):
    extract_dataset_once()
    os.makedirs(FEATURE_FOLDER, exist_ok=True)

    json_path = os.path.join(EXTRACTED_FOLDER, "nsynth-train", "examples.json")
    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Metadata file {json_path} not found. Ensure correct NSynth dataset structure.") from e

    wav_folder = os.path.join(EXTRACTED_FOLDER, "nsynth-train", "audio")
    piano_keys = [
        k for k, v in metadata.items()
        if any(kw in v.get("instrument_family_str", "").lower() for kw in ("keyboard", "piano"))
    ][:max_samples]

    print(f"Found {len(piano_keys)} piano samples.")
    saved_files = []

    for i in tqdm(range(0, len(piano_keys), BATCH_SIZE), desc="Precomputing batches"):
        batch_keys = piano_keys[i:i+BATCH_SIZE]
        audio_list, sr_list, labels = [], [], []

        for key in batch_keys:
            feature_file = os.path.join(FEATURE_FOLDER, f"{key}.npy")
            if os.path.exists(feature_file):
                saved_files.append((feature_file, metadata[key]["pitch"]-21))
                continue
            wav_path = os.path.join(wav_folder, f"{key}.wav")
            try:
                y_audio, sr = sf.read(wav_path)
                y_audio = np.mean(y_audio, axis=1) if y_audio.ndim > 1 else y_audio
                audio_list.append(y_audio)
                sr_list.append(sr)
                labels.append(metadata[key]["pitch"]-21)
            except Exception as e:
                print(f"Skipped {key}: {e}")

        if audio_list:
            feats = extract_features_batch(audio_list, sr_list)
            for feat, key, label in zip(feats, batch_keys, labels):
                np.save(os.path.join(FEATURE_FOLDER, f"{key}.npy"), feat)
                saved_files.append((os.path.join(FEATURE_FOLDER, f"{key}.npy"), label))

    return saved_files

# -------------------------------
# Load precomputed features
# -------------------------------
def load_features(saved_files):
    X, y = [], []
    for feat_path, label in saved_files:
        X.append(np.load(feat_path))
        y.append(label)
    return np.vstack(X), np.array(y)

# -------------------------------
# Training
# -------------------------------
def train(max_samples):
    saved_files = precompute_features(max_samples)
    X, y = load_features(saved_files)
    print(f"Collected {len(y)} samples. Training model…")

    # Debug: Print initial class distribution
    from collections import Counter
    class_counts = Counter(y)
    print("Initial pitch distribution:", {f"MIDI {k+21}": v for k, v in class_counts.items()})

    # Filter classes with fewer than 2 samples
    valid_classes = [cls for cls, count in class_counts.items() if count >= 2]
    valid_indices = [i for i, label in enumerate(y) if label in valid_classes]
    X = X[valid_indices]
    y = y[valid_indices]
    num_samples = len(y)
    num_classes = len(np.unique(y))
    print(f"After filtering, {num_samples} samples remain with {num_classes} classes.")
    print("Filtered pitch distribution:", {f"MIDI {k+21}": v for k, v in Counter(y).items()})

    # Remap labels to contiguous indices (0 to num_classes-1)
    unique_labels = np.unique(y)
    label_to_index = {int(label): idx for idx, label in enumerate(unique_labels)}  # Convert keys to Python int
    y_remapped = np.array([label_to_index[int(label)] for label in y])
    print("Remapped labels:", {f"MIDI {k+21}": v for k, v in zip(unique_labels, range(len(unique_labels)))})

    # Determine test_size for stratification
    if num_samples < 10 or num_classes == 0:
        print("Too few samples or classes after filtering. Using non-stratified split.")
        test_size = 0.2
        stratify = None
    else:
        min_test_size = num_classes / num_samples  # Ensure at least one sample per class
        if min_test_size > 0.5:
            print(f"Minimum test size {min_test_size:.2f} too large. Using non-stratified split.")
            test_size = 0.2
            stratify = None
        else:
            test_size = max(0.2, min_test_size)
            stratify = y_remapped
            print(f"Using stratified split with test_size={test_size:.2f}.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_remapped, test_size=test_size, stratify=stratify, random_state=42
    )

    # Convert to PyTorch tensors and move to GPU
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    # Initialize model
    input_dim = X.shape[1]  # 2 * N_MELS (e.g., 256 for N_MELS=128)
    model = PitchClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    batch_size = 64
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_test).float().mean().item()
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test.cpu().numpy(), preds.cpu().numpy(), zero_division=0))

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_to_index': label_to_index,
        'num_classes': int(num_classes)  # Convert to Python int
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_remapped, test_size=test_size, stratify=stratify, random_state=42
    )

    # Convert to PyTorch tensors and move to GPU
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.long, device=device)
    y_test = torch.tensor(y_test, dtype=torch.long, device=device)

    # Initialize model
    input_dim = X.shape[1]  # 2 * N_MELS (e.g., 256 for N_MELS=128)
    model = PitchClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    batch_size = 64
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == y_test).float().mean().item()
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test.cpu().numpy(), preds.cpu().numpy()))

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_to_index': label_to_index,
        'num_classes': num_classes
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# -------------------------------
# Prediction
# -------------------------------

def predict(wav_file):
    if not os.path.exists(MODEL_PATH):
        print("No model found. Train first.")
        return
    try:
        y_audio, sr = sf.read(wav_file)
        if len(y_audio) < SAMPLE_RATE // 4:
            print("Audio file is too short.")
            return
        if y_audio.ndim > 1:
            y_audio = np.mean(y_audio, axis=1)
        # Resample to 16,000 Hz
        if sr != SAMPLE_RATE:
            y_audio = torchaudio.functional.resample(
                torch.tensor(y_audio).float(), orig_freq=sr, new_freq=SAMPLE_RATE
            ).numpy()
            sr = SAMPLE_RATE
    except Exception as e:
        print(f"Failed to read {wav_file}: {e}")
        return

    feats = extract_features_batch([y_audio], [sr])
    feats = torch.tensor(feats, dtype=torch.float32, device=device)

    checkpoint = torch.load(MODEL_PATH, weights_only=True)
    num_classes = checkpoint['num_classes']
    label_to_index = checkpoint['label_to_index']
    index_to_label = {v: k for k, v in label_to_index.items()}

    model = PitchClassifier(feats.shape[1], num_classes).to(device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    model.eval()

    with torch.no_grad():
        output = model(feats)
        pred_index = torch.argmax(output, dim=1).item()
    pred_label = index_to_label[pred_index]
    midi = pred_label + 21
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    name, octave = notes[midi % 12], midi // 12 - 1

    print(f"Predicted note: {name}{octave} (MIDI {midi})")

# -------------------------------
# Main
# -------------------------------
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
