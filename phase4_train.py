import csv
import os
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from phase3_model import PianoTranscriptionCNN


#                                              
#  DATASET
#                                              

class PianoDataset(Dataset):
    # Dataset for piano transcription. Each sample is a (N_MELS, N_FRAMES) mel spectrogram window extracted from the audio, and a binary label vector of shape (88,) indicating which MIDI pitches are active in that window. The HDF5 file has two resizable datasets: 'windows' of shape (n_samples, N_MELS, N_FRAMES) and 'labels' of shape (n_samples, 88).

    def __init__(self, h5_path: str):
        self.f = h5py.File(h5_path, "r")
        self.windows = self.f["windows"]   # (N, N_MELS, FRAMES)
        self.labels  = self.f["labels"]    # (N, 88)
        print(f"  Loaded {len(self.windows):,} samples from {h5_path}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Add channel dim to (1, N_MELS, FRAMES)
        x = torch.from_numpy(self.windows[idx][None, ...])
        y = torch.from_numpy(self.labels[idx])
        return x, y

    def __del__(self):
        self.f.close()


#                                              
#  METRICS
#                                              

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    # Apply threshold to probabilities to get binary predictions
    y_pred = (y_prob >= threshold).astype(np.int32)

    # Flatten for micro average
    p = precision_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    r = recall_score   (y_true.ravel(), y_pred.ravel(), zero_division=0)
    f = f1_score       (y_true.ravel(), y_pred.ravel(), zero_division=0)
    return p, r, f


#                                              
#  TRAINING LOOP
#                                              


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="  Train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    for x, y in tqdm(loader, desc="  Val  ", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(x)

        # Apply sigmoid — model outputs raw logits now
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    avg_loss   = total_loss / len(loader.dataset)
    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    #    Threshold sweep — find the threshold that maximises F1               
    best_t, best_f1 = 0.5, 0.0
    print("  Threshold sweep:", end="")
    for t in np.arange(0.10, 0.71, 0.05):
        _, _, f = compute_metrics(all_labels, all_probs, threshold=t)
        print(f" {t:.2f}to{f:.3f}", end="")
        if f > best_f1:
            best_f1 = f
            best_t  = t
    print(f"  ← best t={best_t:.2f}")

    p, r, f1 = compute_metrics(all_labels, all_probs, threshold=best_t)
    return avg_loss, p, r, f1, best_t


#                                              
#  CHECKPOINT HELPERS
#                                              

def save_last_checkpoint(path, epoch, model, optimizer, scheduler,
                         best_f1, n_mels, n_frames, history):
    """
    Save a full resume checkpoint after every epoch.
    Overwrites the previous one — only the most recent epoch is kept.
    """
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "scheduler":      scheduler.state_dict(),
        "best_f1":        best_f1,
        "n_mels":         n_mels,
        "n_frames":       n_frames,
        "history":        history,
    }, path)


def save_best_model(path, epoch, model, optimizer, f1, n_mels, n_frames):
    """
    Save best_model.pt whenever val F1 improves.
    Keeps the same format expected by phase5_transcribe.py and live_detector.py.
    """
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "f1":          f1,
        "n_mels":      n_mels,
        "n_frames":    n_frames,
    }, path)


def load_last_checkpoint(path, model, optimizer, scheduler, device):
    """
    Load last_checkpoint.pt and restore all state in-place.

    Returns (start_epoch, best_f1, history) where start_epoch is the
    first epoch that still needs to be trained.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = ckpt["epoch"] + 1          # resume from next epoch
    best_f1     = ckpt["best_f1"]
    history     = ckpt.get("history", [])

    return start_epoch, best_f1, history


def append_history_csv(hist_path: str, row: dict, write_header: bool):
    """Append a single epoch row to the CSV immediately after each epoch."""
    with open(hist_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


#                                              
#  MAIN
#                                              

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥  Device: {device}")

    #    Datasets & loaders                                                     
    print("\nLoading datasets ...")
    train_ds = PianoDataset(args.train_h5)
    val_ds   = PianoDataset(args.val_h5)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Infer shape from dataset
    sample_x, _ = train_ds[0]
    _, n_mels, n_frames = sample_x.shape

    #    Model, loss, optimizer, scheduler                                     
    model = PianoTranscriptionCNN(n_mels=n_mels, n_frames=n_frames).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([3.0]).to(device)
        # pos_weight=3.0 means missing an active note is penalised 3× more
        # than a false positive. Directly increases recall on sparse datasets.
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    os.makedirs(args.output_dir, exist_ok=True)
    last_ckpt_path = os.path.join(args.output_dir, "last_checkpoint.pt")
    best_ckpt_path = os.path.join(args.output_dir, "best_model.pt")
    hist_path      = os.path.join(args.output_dir, "training_history.csv")

    #    Resume or fresh start                                                  
    start_epoch = 1
    best_f1        = 0.0
    best_threshold = 0.5   # updated each epoch by the threshold sweep
    history        = []

    if not args.restart and os.path.exists(last_ckpt_path):
        print(f"\n Resuming from checkpoint: {last_ckpt_path}")
        start_epoch, best_f1, history = load_last_checkpoint(
            last_ckpt_path, model, optimizer, scheduler, device
        )
        best_threshold = history[-1].get("best_threshold", 0.5) if history else 0.5
        print(f"   Resumed at epoch {start_epoch}  (best F1 so far: {best_f1:.3f}  threshold: {best_threshold:.2f})")
    else:
        if args.restart and os.path.exists(last_ckpt_path):
            print(f"\n⚠  --restart passed: ignoring existing checkpoint.")
        print(f"\n Starting fresh training run.")

    end_epoch = start_epoch + args.epochs - 1   # honour --epochs as "epochs to run"

    # If resuming, the CSV already has rows — don't overwrite; append only
    csv_needs_header = not os.path.exists(hist_path) or args.restart
    if args.restart and os.path.exists(hist_path):
        os.remove(hist_path)                     # clear stale history on fresh start

    print(f"\n Training epochs {start_epoch} to {end_epoch} ...\n")

    for epoch in range(start_epoch, end_epoch + 1):
        print(f"Epoch {epoch}/{end_epoch}")

        train_loss                  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, p, r, f1, best_t = evaluate(model, val_loader, criterion, device)
        best_threshold = best_t

        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
              f"  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  best_t={best_t:.2f}  lr={lr_now:.6f}")

        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                   precision=p, recall=r, f1=f1, best_threshold=best_t)
        history.append(row)

        #    Save last_checkpoint.pt (every epoch, for crash recovery)      
        save_last_checkpoint(last_ckpt_path, epoch, model, optimizer, scheduler,
                             best_f1, n_mels, n_frames, history)
        print(f"   last_checkpoint.pt updated  (epoch {epoch})")

        #    Append this epoch to CSV immediately                            
        append_history_csv(hist_path, row, write_header=csv_needs_header)
        csv_needs_header = False                 # only write header once

        #    Save best_model.pt if F1 improved                              
        if f1 > best_f1:
            best_f1 = f1
            save_best_model(best_ckpt_path, epoch, model, optimizer, f1, n_mels, n_frames)
            # Also save best threshold so app.py can use it directly
            torch.save({"threshold": best_t},
                       os.path.join(args.output_dir, "best_threshold.pt"))
            print(f"  💾 best_model.pt updated  (F1={best_f1:.3f}  threshold={best_t:.2f}) to {best_ckpt_path}")

    print(f"\n✅ Training complete!  Best F1 = {best_f1:.3f}")
    print(f"   History  to {hist_path}")
    print(f"   Best     to {best_ckpt_path}")
    print(f"   Resume   to {last_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_h5",   required=True,        help="Path to train HDF5 file")
    parser.add_argument("--val_h5",     required=True,        help="Path to validation HDF5 file")
    parser.add_argument("--epochs",     type=int,   default=30,
                        help="Number of epochs to train in this run "
                             "(added on top of already-completed epochs when resuming)")
    parser.add_argument("--batch",      type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--restart",    action="store_true",
                        help="Ignore any existing checkpoint and train from scratch")
    args = parser.parse_args()
    main(args)
