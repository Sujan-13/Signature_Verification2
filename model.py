import numpy as nn
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)  # ✅ 2 channels
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]   # ✅ Max
        avg_pool = torch.mean(x, dim=1, keepdim=True)     # ✅ Average
        combined = torch.cat([max_pool, avg_pool], dim=1) # ✅ Concatenate
        attention = self.sigmoid(self.conv(combined))
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
            SpatialAttention(),
            nn.Dropout2d(0.1)
        )
        
        # Block 2: 64×64 → 32×32
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            SpatialAttention(),
            nn.Dropout2d(0.15)
        )
        
        # Block 3: 32×32 → 16×16
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            SpatialAttention(),
            nn.Dropout2d(0.2)
        )
        
        # Block 4: 16×16 → 8×8
        self.pool3 = nn.MaxPool2d(2)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            SpatialAttention(),
            nn.Dropout2d(0.2)
        )
        
        # Global pooling: 8×8 → 1×1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256)
        )
    
    # ← CRITICAL: These methods must be at the same indentation level as __init__
    def forward_one(self, x):
        """Forward pass for a single image"""
        # Forward through blocks
        x = self.block1(x)      # 64×64×32
        x = self.pool1(x)       # 32×32×32
        
        x = self.block2(x)      # 32×32×64
        x = self.pool2(x)       # 16×16×64
        
        x = self.block3(x)      # 16×16×128
        x = self.pool3(x)       # 8×8×128
        
        x = self.block4(x)      # 8×8×256
        
        x = self.global_pool(x) # 1×1×256
        x = torch.flatten(x, 1) # 256
        
        x = self.embedding(x)   # embedding_dim
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x
    
    def forward(self, x1, x2=None):
        """
        Forward pass for Siamese network
        
        Args:
            x1: First image tensor
            x2: Optional second image tensor
            
        Returns:
            If x2 is None: returns emb1
            If x2 is provided: returns (emb1, emb2)
        """
        emb1 = self.forward_one(x1)
        if x2 is not None:
            emb2 = self.forward_one(x2)
            return emb1, emb2
        return emb1

init_model=SiameseCNN()
model_compile = torch.compile(init_model, backend="aot_eager", mode="reduce-overhead")   # or "default"
model_final = model_compile.to(device)