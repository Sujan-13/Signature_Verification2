#dataset.py
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
# train_transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.Pad(8, fill=255, padding_mode='constant'),
#     transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.92, 1.08), fill=255),
#     transforms.RandomPerspective(distortion_scale=0.1, p=0.25, fill=255),
#     transforms.RandomApply(
#         [transforms.GaussianBlur(3, sigma=(0.1, 0.5))],
#         p=0.3
#     ),
#     transforms.Resize((64, 64)),
#     transforms.RandomInvert(p=1.0),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,)),
#     # transforms.RandomErasing(p=0.05, scale=(0.02,0.05), ratio=(0.3,3.3))  
# ])
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Pad(8, fill=255, padding_mode='constant'),
    transforms.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.98, 1.1), fill=255),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.25, fill=255),
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
    transforms.Grayscale(num_output_channels=1),
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

def generate_perfectly_balanced_pairs(author_set):
    """
    Generate PERFECTLY balanced pairs with:
    - 50% genuine, 50% forged (EXACT per author, then combined)
    - Skilled forgeries: ONLY same author
    - Random forgeries: ONLY different authors
    """
    pairs = []
    labels = []
    
    stats = {
        'genuine_pairs': 0,
        'skilled_forgeries': 0,
        'random_forgeries': 0,
        'authors_with_no_skilled': 0
    }
    
    all_available_authors = list(author_set)
    
    for author_id in author_set:
        genuine_sigs = genuine_by_author.get(author_id, [])
        forged_sigs = forgery_by_author.get(author_id, [])
        num_genuine = len(genuine_sigs)
        
        if num_genuine == 0:
            continue
        
        # ─── STEP 1: Generate ALL genuine pairs for this author ─────
        genuine_pairs_for_author = []
        
        if num_genuine >= 2:
            for i in range(num_genuine):
                for j in range(i + 1, num_genuine):
                    genuine_pairs_for_author.append((genuine_sigs[i], genuine_sigs[j]))
        elif num_genuine == 1:
            genuine_pairs_for_author.append((genuine_sigs[0], genuine_sigs[0]))
        
        num_genuine_pairs = len(genuine_pairs_for_author)
        
        # ─── STEP 2: Generate skilled forgery pool (SAME AUTHOR ONLY) ───
        skilled_forgery_pool = []
        
        if len(forged_sigs) > 0:
            for genuine_sig in genuine_sigs:
                for forged_sig in forged_sigs:
                    skilled_forgery_pool.append((genuine_sig, forged_sig))
        
        # ─── STEP 3: Generate random forgery pool (OTHER AUTHORS ONLY) ──
        random_forgery_pool = []
        other_authors = [a for a in all_available_authors if a != author_id]
        
        if len(other_authors) > 0:
            num_authors_to_sample = min(
                len(other_authors),
                max(15, num_genuine * 3)  # Sample MORE authors
            )
            
            sampled_other_authors = random.sample(other_authors, num_authors_to_sample)
            
            for genuine_sig in genuine_sigs:
                for other_author in sampled_other_authors:
                    other_sigs = genuine_by_author[other_author]
                    num_sigs_per_author = min(3, len(other_sigs))
                    sampled_other_sigs = random.sample(other_sigs, num_sigs_per_author)
                    
                    for other_sig in sampled_other_sigs:
                        random_forgery_pool.append((genuine_sig, other_sig))
        
        # ─── STEP 4: Balance forgeries for THIS AUTHOR ──────────────
        target_total_forgeries = num_genuine_pairs
        
        # Determine skilled/random split
        if len(skilled_forgery_pool) > 0:
            # Has skilled forgeries - aim for 50/50 split
            target_skilled = target_total_forgeries // 2
            target_random = target_total_forgeries - target_skilled
        else:
            # No skilled forgeries - use ALL random
            target_skilled = 0
            target_random = target_total_forgeries
            stats['authors_with_no_skilled'] += 1
        
        # Sample SKILLED forgeries (WITH replacement if needed, but ONLY from this author)
        sampled_skilled = []
        if target_skilled > 0 and len(skilled_forgery_pool) > 0:
            if len(skilled_forgery_pool) >= target_skilled:
                # Enough skilled - sample without replacement
                sampled_skilled = random.sample(skilled_forgery_pool, target_skilled)
            else:
                # NOT enough skilled - oversample THIS AUTHOR'S skilled forgeries
                sampled_skilled = random.choices(skilled_forgery_pool, k=target_skilled)
                # ↑ This samples WITH replacement from SAME AUTHOR ONLY ✓
        
        # Sample RANDOM forgeries (WITH replacement if needed, but ONLY from other authors)
        sampled_random = []
        if target_random > 0:
            if len(random_forgery_pool) >= target_random:
                # Enough random - sample without replacement
                sampled_random = random.sample(random_forgery_pool, target_random)
            elif len(random_forgery_pool) > 0:
                # NOT enough random - oversample from other authors
                sampled_random = random.choices(random_forgery_pool, k=target_random)
                # ↑ This samples WITH replacement from OTHER AUTHORS ONLY ✓
            else:
                # No random forgeries available - fill with skilled
                if len(skilled_forgery_pool) > 0:
                    extra_skilled = random.choices(skilled_forgery_pool, k=target_random)
                    sampled_skilled.extend(extra_skilled)
        
        # Combine
        final_forgeries = sampled_skilled + sampled_random
        
        # Verify balance for this author
        if len(final_forgeries) != num_genuine_pairs:
            shortage = num_genuine_pairs - len(final_forgeries)
            # Fill shortage with whatever is available
            if len(skilled_forgery_pool) > 0:
                extra = random.choices(skilled_forgery_pool, k=shortage)
            elif len(random_forgery_pool) > 0:
                extra = random.choices(random_forgery_pool, k=shortage)
            else:
                print(f"⚠️ Author {author_id}: Cannot generate enough forgeries!")
                continue
            final_forgeries.extend(extra)
        
        # Add to dataset
        pairs.extend(genuine_pairs_for_author)
        labels.extend([1] * num_genuine_pairs)
        pairs.extend(final_forgeries)
        labels.extend([0] * len(final_forgeries))
        
        # Update stats
        stats['genuine_pairs'] += num_genuine_pairs
        stats['skilled_forgeries'] += len([p for p in final_forgeries if p in skilled_forgery_pool])
        stats['random_forgeries'] += len([p for p in final_forgeries if p in random_forgery_pool])
    
    # Print statistics
    total_forgeries = stats['skilled_forgeries'] + stats['random_forgeries']
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total genuine pairs: {stats['genuine_pairs']}")
    print(f"Total skilled forgeries: {stats['skilled_forgeries']}")
    print(f"Total random forgeries: {stats['random_forgeries']}")
    print(f"Total forgeries: {total_forgeries}")
    print(f"Authors with no skilled forgeries: {stats['authors_with_no_skilled']}")
    print(f"\nBalance ratio: {total_forgeries/stats['genuine_pairs']:.4f}")
    
    if total_forgeries > 0:
        genuine_pct = stats['genuine_pairs'] / (stats['genuine_pairs'] + total_forgeries) * 100
        forged_pct = total_forgeries / (stats['genuine_pairs'] + total_forgeries) * 100
        skilled_pct = stats['skilled_forgeries'] / total_forgeries * 100
        random_pct = stats['random_forgeries'] / total_forgeries * 100
        
        print(f"Positive (genuine): {stats['genuine_pairs']} ({genuine_pct:.1f}%)")
        print(f"Negative (forged):  {total_forgeries} ({forged_pct:.1f}%)")
        print(f"  ├─ Skilled (same author): {stats['skilled_forgeries']} ({skilled_pct:.1f}%)")
        print(f"  └─ Random (other authors): {stats['random_forgeries']} ({random_pct:.1f}%)")
    
    print("="*60)
    
    return pairs, labels


# ═══════════════════════════════════════════════════════════════════════
# GENERATE DATASETS
# ═══════════════════════════════════════════════════════════════════════

print("\n🔄 Generating TRAINING pairs...")
train_pairs, train_labels = generate_perfectly_balanced_pairs(train_authors)

print("\n🔄 Generating VALIDATION pairs...")
val_pairs, val_labels = generate_perfectly_balanced_pairs(val_authors)

print("\n🔄 Generating TEST pairs...")
test_pairs, test_labels = generate_perfectly_balanced_pairs(test_authors)

# Verification
print("\n" + "="*60)
print("FINAL DATASET SUMMARY")
print("="*60)

for split_name, split_labels in [("TRAIN", train_labels), 
                                   ("VAL", val_labels), 
                                   ("TEST", test_labels)]:
    if len(split_labels) == 0:
        print(f"\n{split_name}: ⚠️ EMPTY!")
        continue
        
    total = len(split_labels)
    positive = sum(split_labels)
    negative = total - positive
    
    print(f"\n{split_name}: {total:,} total pairs")
    print(f"  Positive: {positive:,} ({positive/total*100:.1f}%)")
    print(f"  Negative: {negative:,} ({negative/total*100:.1f}%)")

print("="*60)

# Rest of dataloader code...
train_dataset = PairDataset(train_pairs, train_labels, image_cache, transform=train_transform)
val_dataset = PairDataset(val_pairs, val_labels, image_cache, transform=test_transform)
test_dataset = PairDataset(test_pairs, test_labels, image_cache, transform=test_transform)
train_labels = train_dataset.labels
val_labels   = val_dataset.labels
test_labels  = test_dataset.labels
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, 
                               num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                            num_workers=6, pin_memory=True, prefetch_factor=2, persistent_workers=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             num_workers=6, pin_memory=True, prefetch_factor=2, persistent_workers=True)

print(f"\n✓ Dataloaders created")
print(f"  Train: {len(train_dataloader)} batches")
print(f"  Val:   {len(val_dataloader)} batches")
print(f"  Test:  {len(test_dataloader)} batches")