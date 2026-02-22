import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2 
import numpy as np
import random  # For seeding if needed
from torch.utils.data import Dataset, DataLoader

genuine_extract_path = './content/testfin/train_sig/full_org'
forgery_extract_path = './content/testfin/train_sig/full_forg'
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

def generate_balanced_pairs(author_set):
    """
    Generate balanced training pairs with equal genuine and forged samples
    """
    pairs = []
    labels = []
    
    stats = {
        'genuine_pairs': 0,
        'skilled_forgeries': 0,
        'random_forgeries': 0
    }
    
    for author_id in author_set:
        genuine_sigs = genuine_by_author.get(author_id, [])
        forged_sigs = forgery_by_author.get(author_id, [])
        num_genuine = len(genuine_sigs)
        
        # ─── GENUINE-GENUINE PAIRS (POSITIVE) ───────────────────────
        genuine_pairs_for_author = []
        
        if num_genuine >= 2:
            # Generate all combinations
            for i in range(num_genuine):
                for j in range(i + 1, num_genuine):
                    genuine_pairs_for_author.append((genuine_sigs[i], genuine_sigs[j]))
        elif num_genuine == 1:
            # Self-pair as fallback
            genuine_pairs_for_author.append((genuine_sigs[0], genuine_sigs[0]))
            print(f"⚠️ Author {author_id} has only 1 genuine → using self-pair")
        
        num_genuine_pairs = len(genuine_pairs_for_author)
        pairs.extend(genuine_pairs_for_author)
        labels.extend([1] * num_genuine_pairs)
        stats['genuine_pairs'] += num_genuine_pairs
        
        # ─── SKILLED FORGERIES (NEGATIVE) ────────────────────────────
        # Limit genuine signatures to avoid explosion
        max_genuine_for_forgery = min(len(genuine_sigs), 10)
        
        skilled_forgery_pairs = []
        for genuine_sig in genuine_sigs[:max_genuine_for_forgery]:
            for forged_sig in forged_sigs:
                skilled_forgery_pairs.append((genuine_sig, forged_sig))
        
        # ─── RANDOM FORGERIES (NEGATIVE) ──────────────────────────────
        # Sample other authors for random forgeries
        other_authors = [a for a in genuine_by_author.keys() if a != author_id]
        
        random_forgery_pairs = []
        if len(other_authors) > 0:
            # Sample up to 5 other authors
            num_other_authors = min(5, len(other_authors))
            sampled_authors = random.sample(other_authors, k=num_other_authors)
            
            for genuine_sig in genuine_sigs[:max_genuine_for_forgery]:
                for other_author in sampled_authors:
                    other_sigs = genuine_by_author[other_author]
                    # Sample up to 2 signatures from each other author
                    for other_sig in random.sample(other_sigs, min(2, len(other_sigs))):
                        random_forgery_pairs.append((genuine_sig, other_sig))
        
        # ─── BALANCE FORGERIES TO MATCH GENUINE PAIRS ─────────────────
        # We want: num_forgeries ≈ num_genuine_pairs
        total_forgery_pairs = skilled_forgery_pairs + random_forgery_pairs
        
        # If too many forgeries, sample down to match genuine pairs
        target_forgeries = num_genuine_pairs
        
        if len(total_forgery_pairs) > target_forgeries:
            # Ensure we keep some of both types
            min_skilled = min(len(skilled_forgery_pairs), target_forgeries // 2)
            min_random = target_forgeries - min_skilled
            
            sampled_skilled = random.sample(skilled_forgery_pairs, 
                                          min(min_skilled, len(skilled_forgery_pairs)))
            sampled_random = random.sample(random_forgery_pairs, 
                                         min(min_random, len(random_forgery_pairs)))
            
            final_forgeries = sampled_skilled + sampled_random
        else:
            final_forgeries = total_forgery_pairs
        
        # Add forgeries to dataset
        pairs.extend(final_forgeries)
        labels.extend([0] * len(final_forgeries))
        
        # Update statistics
        num_skilled = sum(1 for p in final_forgeries if p in skilled_forgery_pairs)
        num_random = len(final_forgeries) - num_skilled
        
        stats['skilled_forgeries'] += num_skilled
        stats['random_forgeries'] += num_random
        
        # Print per-author stats
        if author_id == list(author_set)[0]:  # Print for first author as example
            print(f"\nExample (Author {author_id}):")
            print(f"  Genuine pairs: {num_genuine_pairs}")
            print(f"  Skilled forgeries: {num_skilled}")
            print(f"  Random forgeries: {num_random}")
            print(f"  Total forgeries: {len(final_forgeries)}")
            print(f"  Balance ratio: {len(final_forgeries)/num_genuine_pairs:.2f}")
    
    # Print overall statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total genuine pairs: {stats['genuine_pairs']}")
    print(f"Total skilled forgeries: {stats['skilled_forgeries']}")
    print(f"Total random forgeries: {stats['random_forgeries']}")
    total_forgeries = stats['skilled_forgeries'] + stats['random_forgeries']
    print(f"Total forgeries: {total_forgeries}")
    print(f"\nBalance ratio (forgeries/genuine): {total_forgeries/stats['genuine_pairs']:.2f}")
    print(f"Positive samples: {stats['genuine_pairs']} ({stats['genuine_pairs']/(stats['genuine_pairs']+total_forgeries)*100:.1f}%)")
    print(f"Negative samples: {total_forgeries} ({total_forgeries/(stats['genuine_pairs']+total_forgeries)*100:.1f}%)")
    print("="*60)
    
    return pairs, labels


# ═══════════════════════════════════════════════════════════════════════
# USAGE - Replace your existing generate_pairs function with this
# ═══════════════════════════════════════════════════════════════════════

# Generate balanced datasets
print("\n🔄 Generating TRAINING pairs...")
train_pairs, train_labels = generate_balanced_pairs(train_authors)

print("\n🔄 Generating VALIDATION pairs...")
val_pairs, val_labels = generate_balanced_pairs(val_authors)

print("\n🔄 Generating TEST pairs...")
test_pairs, test_labels = generate_balanced_pairs(test_authors)

# Verify balance
print("\n" + "="*60)
print("FINAL DATASET SUMMARY")
print("="*60)
print(f"\nTRAIN: {len(train_labels)} total pairs")
print(f"  Positive: {sum(train_labels)} ({sum(train_labels)/len(train_labels)*100:.1f}%)")
print(f"  Negative: {len(train_labels)-sum(train_labels)} ({(len(train_labels)-sum(train_labels))/len(train_labels)*100:.1f}%)")

print(f"\nVAL: {len(val_labels)} total pairs")
print(f"  Positive: {sum(val_labels)} ({sum(val_labels)/len(val_labels)*100:.1f}%)")
print(f"  Negative: {len(val_labels)-sum(val_labels)} ({(len(val_labels)-sum(val_labels))/len(val_labels)*100:.1f}%)")

print(f"\nTEST: {len(test_labels)} total pairs")
print(f"  Positive: {sum(test_labels)} ({sum(test_labels)/len(test_labels)*100:.1f}%)")
print(f"  Negative: {len(test_labels)-sum(test_labels)} ({(len(test_labels)-sum(test_labels))/len(test_labels)*100:.1f}%)")



test_pairs, test_labels = generate_balanced_pairs(test_authors)
train_pairs, train_labels = generate_balanced_pairs(train_authors)
val_pairs, val_labels = generate_balanced_pairs(val_authors)

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