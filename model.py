import numpy as nn
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Generate attention map from max and avg pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        attention = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention

class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()
        
        # Block 1: 64×64 → 64×64 (no pooling!)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout2d(0.1)
        )
        
        # Block 2: 64×64 → 32×32
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Stride instead of pool
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.1)
        )
        
        # Block 3: 32×32 → 16×16
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Dropout2d(0.15)
        )
        
        # Block 4: 16×16 → 16×16 (keep spatial info!)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Dropout2d(0.15)
        )
        
        # Global pooling: 16×16 → 1×1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding
        self.embedding = nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
    
    def forward_one(self, x):
        x = self.block1(x)      # 64×64
        x = self.block2(x)      # 32×32
        x = self.block3(x)      # 16×16
        x = self.block4(x)      # 16×16 (preserved!)
        x = self.global_pool(x) # 1×1
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, x1, x2=None):
        emb1 = self.forward_one(x1)
        if x2 is not None:
            emb2 = self.forward_one(x2)
            return emb1, emb2
        return emb1

init_model=SiameseCNN()
model_compile = torch.compile(init_model, backend="aot_eager", mode="reduce-overhead")   # or "default"
model_final = model_compile.to(device)