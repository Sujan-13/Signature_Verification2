# dataset.py
import os
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & SEEDS
# ─────────────────────────────────────────────────────────────────────────────
genuine_extract_path = './content/testfin/train_sig/full_org'
forgery_extract_path = './content/testfin/train_sig/full_forg'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Pad(16, fill=255, padding_mode='constant'),
    transforms.RandomAffine(
        degrees=15, translate=(0.08, 0.08), scale=(0.98, 1.1), fill=255),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.25, fill=255),
    transforms.RandomApply(
        [transforms.GaussianBlur(3, sigma=(0.1, 0.5))], p=0.3),
    transforms.Resize((96, 96)),
    transforms.RandomInvert(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

test_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomInvert(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ─────────────────────────────────────────────────────────────────────────────
# PAIR DATASET
# ─────────────────────────────────────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, pairs, labels, image_cache, transform=None):
        self.pairs       = pairs
        self.labels      = labels
        self.image_cache = image_cache
        self.transform   = transform

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


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE CACHE & AUTHOR MAPS
# ─────────────────────────────────────────────────────────────────────────────
image_cache       = {}
genuine_by_author = {}
forgery_by_author = {}

for filename in os.listdir(genuine_extract_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.PNG')):
        try:
            author_id = filename.split('_')[1]
            img_path  = os.path.join(genuine_extract_path, filename)
            genuine_by_author.setdefault(author_id, []).append(img_path)
            image_cache[img_path] = Image.open(img_path).convert('L')
        except (ValueError, IndexError):
            print(f"Skipping invalid genuine filename: {filename}")

for filename in os.listdir(forgery_extract_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.PNG')):
        try:
            author_id = filename.split('_')[1]
            img_path  = os.path.join(forgery_extract_path, filename)
            forgery_by_author.setdefault(author_id, []).append(img_path)
            image_cache[img_path] = Image.open(img_path).convert('L')
        except (ValueError, IndexError):
            print(f"Skipping invalid forgery filename: {filename}")

print(f"Genuine authors : {len(genuine_by_author)}")
print(f"Forgery authors : {len(forgery_by_author)}")
for author_id, sigs in genuine_by_author.items():
    print(f"  Author {author_id}: {len(sigs)} genuine signatures")
for author_id, sigs in forgery_by_author.items():
    print(f"  Author {author_id}: {len(sigs)} forged signatures")
print(f"Cached {len(image_cache)} unique images in RAM")


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
# FIX #1: sort before shuffle so the order is fully deterministic with the seed
all_authors = sorted(genuine_by_author.keys())
random.seed(SEED)
random.shuffle(all_authors)

total_authors = len(all_authors)
train_count   = int(0.8 * total_authors)
val_count     = int(0.1 * total_authors)

train_authors = set(all_authors[:train_count])
val_authors   = set(all_authors[train_count:train_count + val_count])
test_authors  = set(all_authors[train_count + val_count:])

print(f"\nTotal authors : {total_authors}")
print(f"Train authors : {len(train_authors)}")
print(f"Val   authors : {len(val_authors)}")
print(f"Test  authors : {len(test_authors)}")
print(f"Sum           : {len(train_authors)+len(val_authors)+len(test_authors)}")


# ─────────────────────────────────────────────────────────────────────────────
# PAIR GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_perfectly_balanced_pairs(author_set):
    """
    Generate perfectly balanced pairs:
      - 50 % genuine, 50 % forged (exact, per author)
      - Skilled forgeries  : same author only
      - Random forgeries   : different authors only

    FIX #1: iterate over sorted(author_set) so results are reproducible
             when the same SEED is used.
    FIX #5: use sets for O(1) pool-membership checks instead of O(n) lists.
    """
    pairs  = []
    labels = []

    stats = {
        'genuine_pairs':          0,
        'skilled_forgeries':      0,
        'random_forgeries':       0,
        'authors_with_no_skilled': 0,
    }

    # FIX #1: sort so iteration order is deterministic
    all_available_authors = sorted(author_set)

    for author_id in sorted(author_set):          # FIX #1 (outer loop too)
        genuine_sigs = genuine_by_author.get(author_id, [])
        forged_sigs  = forgery_by_author.get(author_id, [])
        num_genuine  = len(genuine_sigs)

        if num_genuine == 0:
            continue

        # ── STEP 1: genuine pairs ────────────────────────────────────────
        genuine_pairs_for_author = []
        if num_genuine >= 2:
            for i in range(num_genuine):
                for j in range(i + 1, num_genuine):
                    genuine_pairs_for_author.append(
                        (genuine_sigs[i], genuine_sigs[j]))
        else:
            genuine_pairs_for_author.append((genuine_sigs[0], genuine_sigs[0]))

        num_genuine_pairs = len(genuine_pairs_for_author)

        # ── STEP 2: skilled forgery pool (same author) ───────────────────
        skilled_forgery_pool = []
        if forged_sigs:
            for g in genuine_sigs:
                for f in forged_sigs:
                    skilled_forgery_pool.append((g, f))

        # FIX #5: set for O(1) membership test
        skilled_pool_set = set(skilled_forgery_pool)

        # ── STEP 3: random forgery pool (other authors) ──────────────────
        random_forgery_pool = []
        other_authors = [a for a in all_available_authors if a != author_id]

        if other_authors:
            num_to_sample = min(len(other_authors),
                                max(15, num_genuine * 3))
            sampled_others = random.sample(other_authors, num_to_sample)

            for g in genuine_sigs:
                for other_author in sampled_others:
                    other_sigs = genuine_by_author[other_author]
                    k          = min(3, len(other_sigs))
                    for s in random.sample(other_sigs, k):
                        random_forgery_pool.append((g, s))

        # FIX #5: set for O(1) membership test
        random_pool_set = set(random_forgery_pool)

        # ── STEP 4: balance forgeries ────────────────────────────────────
        target_total = num_genuine_pairs

        if skilled_forgery_pool:
            target_skilled = target_total // 2
            target_random  = target_total - target_skilled
        else:
            target_skilled = 0
            target_random  = target_total
            stats['authors_with_no_skilled'] += 1

        # Sample skilled
        sampled_skilled = []
        if target_skilled > 0 and skilled_forgery_pool:
            if len(skilled_forgery_pool) >= target_skilled:
                sampled_skilled = random.sample(skilled_forgery_pool, target_skilled)
            else:
                sampled_skilled = random.choices(skilled_forgery_pool, k=target_skilled)

        # Sample random
        sampled_random = []
        if target_random > 0:
            if len(random_forgery_pool) >= target_random:
                sampled_random = random.sample(random_forgery_pool, target_random)
            elif random_forgery_pool:
                sampled_random = random.choices(random_forgery_pool, k=target_random)
            elif skilled_forgery_pool:
                extra = random.choices(skilled_forgery_pool, k=target_random)
                sampled_skilled.extend(extra)

        final_forgeries = sampled_skilled + sampled_random

        # Patch any remaining shortage
        if len(final_forgeries) != num_genuine_pairs:
            shortage = num_genuine_pairs - len(final_forgeries)
            pool = skilled_forgery_pool or random_forgery_pool
            if not pool:
                print(f"⚠️  Author {author_id}: cannot generate enough forgeries!")
                continue
            final_forgeries.extend(random.choices(pool, k=shortage))

        # ── Add to dataset ───────────────────────────────────────────────
        pairs.extend(genuine_pairs_for_author)
        labels.extend([1] * num_genuine_pairs)
        pairs.extend(final_forgeries)
        labels.extend([0] * len(final_forgeries))

        # FIX #5: O(1) set lookups instead of O(n) list scan
        stats['genuine_pairs']     += num_genuine_pairs
        stats['skilled_forgeries'] += sum(
            1 for p in final_forgeries if p in skilled_pool_set)
        stats['random_forgeries']  += sum(
            1 for p in final_forgeries if p in random_pool_set)

    # ── Print stats ──────────────────────────────────────────────────────────
    total_forgeries = stats['skilled_forgeries'] + stats['random_forgeries']
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total genuine pairs      : {stats['genuine_pairs']}")
    print(f"Total skilled forgeries  : {stats['skilled_forgeries']}")
    print(f"Total random forgeries   : {stats['random_forgeries']}")
    print(f"Total forgeries          : {total_forgeries}")
    print(f"Authors with no skilled  : {stats['authors_with_no_skilled']}")

    if total_forgeries > 0:
        g_pct  = stats['genuine_pairs']     / (stats['genuine_pairs'] + total_forgeries) * 100
        f_pct  = total_forgeries            / (stats['genuine_pairs'] + total_forgeries) * 100
        sk_pct = stats['skilled_forgeries'] / total_forgeries * 100
        ra_pct = stats['random_forgeries']  / total_forgeries * 100
        print(f"\nBalance ratio            : {total_forgeries/stats['genuine_pairs']:.4f}")
        print(f"Positive (genuine)       : {stats['genuine_pairs']} ({g_pct:.1f}%)")
        print(f"Negative (forged)        : {total_forgeries} ({f_pct:.1f}%)")
        print(f"  ├─ Skilled (same auth) : {stats['skilled_forgeries']} ({sk_pct:.1f}%)")
        print(f"  └─ Random (other auth) : {stats['random_forgeries']} ({ra_pct:.1f}%)")
    print("=" * 60)

    return pairs, labels


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE SPLITS
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔄 Generating TRAINING pairs...")
train_pairs, train_labels = generate_perfectly_balanced_pairs(train_authors)

print("\n🔄 Generating VALIDATION pairs...")
val_pairs, val_labels = generate_perfectly_balanced_pairs(val_authors)

print("\n🔄 Generating TEST pairs...")
test_pairs, test_labels = generate_perfectly_balanced_pairs(test_authors)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)
for split_name, split_labels in [
        ("TRAIN", train_labels),
        ("VAL",   val_labels),
        ("TEST",  test_labels)]:
    if not split_labels:
        print(f"\n{split_name}: ⚠️  EMPTY!")
        continue
    total    = len(split_labels)
    positive = sum(split_labels)
    negative = total - positive
    print(f"\n{split_name}: {total:,} total pairs")
    print(f"  Positive: {positive:,} ({positive/total*100:.1f}%)")
    print(f"  Negative: {negative:,} ({negative/total*100:.1f}%)")
print("=" * 60)