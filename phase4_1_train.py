import csv, os, argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from phase3_nsynth_model import NSynthPitchCNN


#    Dataset                                                                    

class NSynthDataset(Dataset):
    # Dataset for NSynth pitch classification. Each sample is a single note, so the label is a single MIDI pitch class (0-87). The input is a (N_MELS, N_FRAMES) mel spectrogram window extracted from the audio. The HDF5 file has two resizable datasets: 'windows' of shape (n_samples, N_MELS, N_FRAMES) and 'labels' of shape (n_samples,).
    def __init__(self, h5_path: str):
        self.f       = h5py.File(h5_path, 'r')
        self.windows = self.f['windows']   # (N, N_MELS, N_FRAMES)
        self.labels  = self.f['labels']    # (N,) int64 — pitch index 0-87
        print(f'  Loaded {len(self.windows):,} samples from {h5_path}')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx][None, ...])  # (1, N_MELS, N_FRAMES)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y

    def __del__(self):
        self.f.close()


#    Metrics                                                                    

def compute_accuracy(logits: np.ndarray, labels: np.ndarray, top_k=(1, 5)):
    """
    Compute top-1 and top-5 accuracy.
    logits: (N, 88)
    labels: (N,) integer pitch indices
    """
    results = {}
    for k in top_k:
        top_k_preds = np.argsort(logits, axis=1)[:, -k:]
        correct = sum(labels[i] in top_k_preds[i] for i in range(len(labels)))
        results[f'top{k}'] = correct / len(labels)
    return results


#    Training loop                                                              

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc='  Train', leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        # Gradient clipping — prevents rare exploding gradient issues
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)

#   Evaluation loop — computes average loss and top-1/top-5 accuracy on the validation set. No gradient updates, and model is in eval() mode to disable dropout.
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss  = 0.0
    all_logits  = []
    all_labels  = []
    for x, y in tqdm(loader, desc='  Val  ', leave=False):
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss  += loss.item() * len(x)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_labels = np.concatenate(all_labels)
    avg_loss   = total_loss / len(loader.dataset)
    acc        = compute_accuracy(all_logits, all_labels, top_k=(1, 5))
    return avg_loss, acc['top1'], acc['top5']


#    Checkpointing                                                              

def save_checkpoint(path, epoch, model, optimizer, scheduler, best_acc, history, n_mels, n_frames):
    torch.save({
        'epoch':       epoch,
        'model_state': model.state_dict(),
        'optimizer':   optimizer.state_dict(),
        'scheduler':   scheduler.state_dict(),
        'best_acc':    best_acc,
        'history':     history,
        'n_mels':      n_mels,
        'n_frames':    n_frames,
        'model_type':  'nsynth',
    }, path)


def save_best(path, epoch, model, optimizer, acc, n_mels, n_frames):
    torch.save({
        'epoch':       epoch,
        'model_state': model.state_dict(),
        'optimizer':   optimizer.state_dict(),
        'accuracy':    acc,
        'n_mels':      n_mels,
        'n_frames':    n_frames,
        'model_type':  'nsynth',
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt['epoch'] + 1, ckpt['best_acc'], ckpt.get('history', [])


#    Main                                                                       

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n🖥  Device: {device}')

    print('\nLoading datasets …')
    train_ds = NSynthDataset(args.train_h5)
    val_ds   = NSynthDataset(args.val_h5)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=0, pin_memory=True)

    sample_x, _ = train_ds[0]
    _, n_mels, n_frames = sample_x.shape

    model  = NSynthPitchCNN(n_mels=n_mels, n_frames=n_frames).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'NSynthPitchCNN — {params:,} parameters  ({n_mels} mels × {n_frames} frames)')

    # CrossEntropyLoss for single-label classification
    # Label smoothing=0.1 prevents overconfidence and helps generalisation
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # CosineAnnealingLR — smoothly reduces lr to near-zero over training
    # Better than ReduceLROnPlateau for clean classification tasks
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(args.output_dir, exist_ok=True)
    last_ckpt = os.path.join(args.output_dir, 'nsynth_last_checkpoint.pt')
    best_ckpt = os.path.join(args.output_dir, 'nsynth_best_model.pt')
    hist_path = os.path.join(args.output_dir, 'nsynth_training_history.csv')

    start_epoch = 1
    best_acc    = 0.0
    history     = []

    if not args.restart and os.path.exists(last_ckpt):
        print(f'\n Resuming from {last_ckpt}')
        start_epoch, best_acc, history = load_checkpoint(
            last_ckpt, model, optimizer, scheduler, device)
        print(f'   Resumed at epoch {start_epoch}  (best top-1 acc: {best_acc:.3f})')
    else:
        if args.restart and os.path.exists(last_ckpt):
            print('\n⚠  --restart: ignoring existing checkpoint.')
        print('\n Starting fresh NSynth training run.')

    end_epoch = start_epoch + args.epochs - 1
    csv_header = not os.path.exists(hist_path) or args.restart
    if args.restart and os.path.exists(hist_path):
        os.remove(hist_path)

    print(f'\n Training epochs {start_epoch} → {end_epoch} …\n')

    for epoch in range(start_epoch, end_epoch + 1):
        print(f'Epoch {epoch}/{end_epoch}')

        train_loss          = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, top1, top5 = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        print(f'  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}'
              f'  top1={top1:.3f} ({top1*100:.1f}%)'
              f'  top5={top5:.3f} ({top5*100:.1f}%)'
              f'  lr={lr_now:.6f}')

        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                   top1_acc=top1, top5_acc=top5)
        history.append(row)

        # Save last checkpoint every epoch
        save_checkpoint(last_ckpt, epoch, model, optimizer, scheduler,
                        best_acc, history, n_mels, n_frames)

        # Append to CSV
        with open(hist_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if csv_header:
                writer.writeheader()
                csv_header = False
            writer.writerow(row)

        # Save best model
        if top1 > best_acc:
            best_acc = top1
            save_best(best_ckpt, epoch, model, optimizer, top1, n_mels, n_frames)
            print(f'  💾 nsynth_best_model.pt  (top-1={best_acc:.3f} = {best_acc*100:.1f}%)'
                  f'  → {best_ckpt}')

    print(f'\n✅ NSynth training complete!')
    print(f'   Best top-1 accuracy: {best_acc:.3f} ({best_acc*100:.1f}%)')
    print(f'   Model → {best_ckpt}')
    print(f'   History → {hist_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_h5',   required=True)
    parser.add_argument('--val_h5',     required=True)
    parser.add_argument('--output_dir', default='./nsynth_checkpoints')
    parser.add_argument('--epochs',     type=int,   default=30)
    parser.add_argument('--batch',      type=int,   default=128)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--restart',    action='store_true')
    main(parser.parse_args())
