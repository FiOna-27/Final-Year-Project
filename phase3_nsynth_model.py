import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → optional MaxPool."""
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


class NSynthPitchCNN(nn.Module):
    
    #Compact CNN for single-note piano pitch classification.

    #Smaller than the MAESTRO CNN because:
    #- NSynth examples are clean isolated notes (less variation to model)
    #- We only need to classify 88 pitches (not detect multiple simultaneously)
    #- Smaller model = less overfitting on 70,000 clean examples

    
    # Architecture inspired by common audio CNNs, but simplified for this task.
    def __init__(self, n_mels: int = 64, n_frames: int = 32, n_notes: int = 88):
        super().__init__()
        self.n_mels   = n_mels
        self.n_frames = n_frames
        self.n_notes  = n_notes

        self.features = nn.Sequential(
            ConvBlock(1,  32, pool=True),  
            ConvBlock(32, 64, pool=True),   
            ConvBlock(64, 128, pool=False),
            nn.Dropout2d(0.2),               # lighter dropout, clean data
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, n_notes),
            # NO Softmax, CrossEntropyLoss applies it during training.
            # app.py applies torch.softmax() at inference time.
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)   # (B, 128, H, W)
        x = self.pool(x)       # (B, 128, 4, 4)
        x = self.classifier(x) # (B, 88) — raw logits
        return x


#    Quick shape check                                                          
if __name__ == '__main__':
    model = NSynthPitchCNN(n_mels=64, n_frames=32)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'NSynthPitchCNN — {total:,} trainable parameters')

    dummy = torch.zeros(4, 1, 64, 32)
    out   = model(dummy)
    probs = torch.softmax(out, dim=1)
    print(f'Input:  {tuple(dummy.shape)}')
    print(f'Output (logits): {tuple(out.shape)}')
    print(f'After softmax — sums to 1: {probs.sum(dim=1)}')
    assert out.shape == (4, 88)
    print('✅ Shape check passed.')
