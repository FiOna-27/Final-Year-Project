import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    # Simple convolutional block: Conv2d → BatchNorm → ReLU → (optional MaxPool)
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PianoTranscriptionCNN(nn.Module):
    # Compact CNN for single-note piano pitch classification. Smaller than the MAESTRO CNN because:
    # - NSynth examples are clean isolated notes (less variation to model)
    # - We only need to classify 88 pitches (not detect multiple simultaneously)
    # - Smaller model = less overfitting on 70,000 clean examples
    #  Architecture inspired by common audio CNNs, but simplified for this task.
    def __init__(self, n_mels: int = 64, n_frames: int = 11, n_notes: int = 88):
        super().__init__()
        self.n_mels   = n_mels
        self.n_frames = n_frames
        self.n_notes  = n_notes

        # Feature extractor 
        self.features = nn.Sequential(
            ConvBlock(1,  32, pool=True),  
            ConvBlock(32, 64, pool=True),   
            ConvBlock(64, 128, pool=False), 
            nn.Dropout2d(0.25),
        )

        # Adaptive pool collapses spatial dims to (4, 4) regardless of input size because the fully-connected head expects a fixed-size input. 
        # This allows flexibility in N_MELS and N_FRAMES.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier head with two hidden layers and dropout for regularization. Outputs raw logits for BCEWithLogitsLoss.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_notes),
            # NO Sigmoid here, BCEWithLogitsLoss applies it internally during
            # training (numerically more stable). At inference, app.py applies
            # torch.sigmoid() manually before thresholding.
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 1, N_MELS, N_FRAMES)
        x = self.features(x)         # (B, 128, H, W)
        x = self.adaptive_pool(x)    # (B, 128, 4, 4)
        x = self.classifier(x)       # (B, 88)
        return x


#     Quick shape check                                                         
if __name__ == "__main__":
    model = PianoTranscriptionCNN(n_mels=64, n_frames=11)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PianoTranscriptionCNN — {total:,} trainable parameters")

    dummy = torch.zeros(4, 1, 64, 11)   # batch of 4 windows
    out   = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}   (expected: (4, 88)) — raw logits")
    assert out.shape == (4, 88), "Shape mismatch!"
    print("✅ Shape check passed. (Output is raw logits — apply sigmoid for probabilities)")
