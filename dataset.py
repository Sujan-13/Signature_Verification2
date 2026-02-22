import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2 
import numpy as np
import random  # For seeding if needed
from torch.utils.data import Dataset, DataLoader

genuine_extract_path = './content/train_sig/full_org'
forgery_extract_path = './content/train_sig/full_forg'
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# Define the transforms
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Pad(8, fill=255, padding_mode='constant'),
    transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.92, 1.08), fill=255),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.25, fill=255),
    transforms.RandomApply(
        [transforms.GaussianBlur(3, sigma=(0.1, 0.5))],
        p=0.3
    ),
    transforms.Resize((64, 64)),
    transforms.RandomInvert(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    # transforms.RandomErasing(p=0.05, scale=(0.02,0.05), ratio=(0.3,3.3))  
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to match FashionMNIST if needed
    transforms.RandomInvert(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Standard normalization
])

# Custom dataset for pairs (to allow different transforms)
class PairDataset(Dataset):
    def __init__(self, pairs, labels, image_cache, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.image_cache = image_cache
        self.transform = transform
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        img1 = self.image_cache[img1_path].copy()
        img2 = self.image_cache[img2_path].copy()

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# Manually generate pairs, labels, and image_cache (extracted from original class init)
image_cache = {}
genuine_by_author = {}
forgery_by_author = {}

# Load genuine signatures
for filename in os.listdir(genuine_extract_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.PNG')):
        try:
            author_id = filename.split('_')[1]
            img_path = os.path.join(genuine_extract_path, filename)
            if author_id not in genuine_by_author:
                genuine_by_author[author_id] = []
            genuine_by_author[author_id].append(img_path)
            image_cache[img_path] = Image.open(img_path).convert('L')
        except (ValueError, IndexError):
            print(f"Skipping invalid genuine filename: {filename}")
            continue

# Load forged signatures
for filename in os.listdir(forgery_extract_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.PNG')):
        try:
            author_id = filename.split('_')[1]
            img_path = os.path.join(forgery_extract_path, filename)
            if author_id not in forgery_by_author:
                forgery_by_author[author_id] = []
            forgery_by_author[author_id].append(img_path)
            image_cache[img_path] = Image.open(img_path).convert('L')
        except (ValueError, IndexError):
            print(f"Skipping invalid forgery filename: {filename}")
            continue

print(f"Genuine authors: {len(genuine_by_author)}, {genuine_by_author.keys()}")
print(f"Forgery authors: {len(forgery_by_author)}, {forgery_by_author.keys()}")
for author_id, sigs in genuine_by_author.items():
    print(f"Author {author_id}: {len(sigs)} genuine signatures")
for author_id, sigs in forgery_by_author.items():
    print(f"Author {author_id}: {len(sigs)} forged signatures")
print(f"Cached {len(image_cache)} unique images in RAM")



all_authors = list(genuine_by_author.keys())
random.seed(SEED)   # ← Reset seed before shuffle
random.shuffle(all_authors)

total_authors = len(all_authors)
train_count = int(0.8 * total_authors)
val_count   = int(0.1 * total_authors)
test_count  = total_authors - train_count - val_count  # make sure we use all authors

# Create the three sets
train_authors = set(all_authors[:train_count])
val_authors   = set(all_authors[train_count:train_count + val_count])
test_authors  = set(all_authors[train_count + val_count:]) 

print(f"Total authors: {total_authors}")
print(f"Train authors: {len(train_authors)}")
print(f"Val   authors: {len(val_authors)}")   # ← add this
print(f"Test  authors: {len(test_authors)}")
print(f"Train + Val + Test = {len(train_authors) + len(val_authors) + len(test_authors)} (should equal total)")

def generate_pairs(author_set):
    pairs = []
    labels = []

    for author_id in author_set:
        genuine_sigs = genuine_by_author.get(author_id, [])
        forged_sigs = forgery_by_author.get(author_id, [])
        num_genuine = len(genuine_sigs)

        # Genuine-Genuine pairs (positive)
        if num_genuine >= 2:
            # Normal cross pairs
            for i in range(num_genuine):
                for j in range(i + 1, num_genuine):
                    pairs.append((genuine_sigs[i], genuine_sigs[j]))
                    labels.append(1)
        elif num_genuine == 1:
            # Self-pair: same image twice → positive
            img = genuine_sigs[0]
            pairs.append((img, img))
            labels.append(1)
            print(f"Author {author_id} has only 1 genuine → using self-pair")

        # Genuine-Forgery pairs (negative)
        for genuine_sig in genuine_sigs[:12]:
            for forged_sig in forged_sigs:
                pairs.append((genuine_sig, forged_sig))
                labels.append(0)

        other_authors = random.sample([a for a in genuine_by_author if a != author_id], k=min(3, len(genuine_by_author)-1))
        for other_author in other_authors:
            other_genuine_sigs = genuine_by_author[other_author]
            for other_sig in other_genuine_sigs:
                pairs.append((genuine_sig, other_sig))
                labels.append(0)


    return pairs, labels

test_pairs, test_labels = generate_pairs(test_authors)
train_pairs, train_labels = generate_pairs(train_authors)
val_pairs, val_labels = generate_pairs(val_authors)

print(f"Train pairs: {len(train_pairs)}, Train labels: {len(train_labels)}")
print(f"Train positive: {sum(1 for label in train_labels if label ==1)}")
print(f"Train negative: {sum(1 for label in train_labels if label == 0)}")
print(f"Val pairs: {len(val_pairs)}, Val labels: {len(val_labels)}")
print(f"Val positive: {sum(1 for label in val_labels if label == 1)}")
print(f"Val negative: {sum(1 for label in val_labels if label == 0)}")
print(f"Test pairs: {len(test_pairs)}, Test labels: {len(test_labels)}")
print(f"Test positive: {sum(1 for label in test_labels if label == 1)}")
print(f"Test negative: {sum(1 for label in test_labels if label == 0)}")
print(f"Train positive ratio: {sum(l==1 for l in train_labels) / len(train_labels):.3f}")
train_dataset = PairDataset(train_pairs, train_labels, image_cache, transform=train_transform)
val_dataset = PairDataset(val_pairs, val_labels, image_cache, transform=test_transform)
test_dataset = PairDataset(test_pairs, test_labels, image_cache, transform=test_transform)

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=64,           
    shuffle=True,
    num_workers=8,            
    pin_memory=True,          
    prefetch_factor=2,
    persistent_workers=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=128,           
    shuffle=False,
    num_workers=6,            
    pin_memory=True,          
    prefetch_factor=2,
    persistent_workers=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage

# Helper to reverse normalization so we can visualize properly
def denormalize(tensor):
    # Assuming Normalize((0.5,), (0.5,))
    tensor = tensor.clone()  # don't modify original
    tensor = tensor * 0.5 + 0.5  # reverse Normalize
    tensor = tensor.clamp(0, 1)
    return tensor

# Show a grid of 8 random augmented samples
num_samples = 8
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for i in range(num_samples):
    # Get a random sample from train_dataset
    img1, img2, label = train_dataset[random.randint(0, len(train_dataset)-1)]
    
    # We usually want to look at img1 (or img2 — doesn't matter)
    img = denormalize(img1)           # reverse normalization
    img_pil = ToPILImage()(img)       # convert tensor → PIL Image
    
    # Show image
    axes[i].imshow(img_pil, cmap='gray')
    axes[i].set_title(f"Label: {label.item()}\nAugmented sample")
    axes[i].axis('off')

plt.tight_layout()
plt.suptitle("Random samples after train_transform (should look reasonable)", fontsize=14)
plt.show()