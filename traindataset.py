import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2 
import numpy as np

genuine_extract_path = './content/PNG/Offline Genuine'
forgery_extract_path = './content/PNG/Offline Forgeries'
class SignatureVerificationDataset(Dataset):
    def __init__(self, genuine_dir, forgery_dir, transform=None):
        """
        Args:
            genuine_dir (str): Path to offline_genuine folder (e.g., './data/offline_genuine').
            forgery_dir (str): Path to offline_forgeries folder (e.g., './data/offline_forgeries').
            transform (callable, optional): Transformations to apply to images.
        """
        self.genuine_dir = genuine_dir
        self.forgery_dir = forgery_dir
        self.transform = transform
        self.pairs = []
        self.labels = []
        self.image_cache = {}
        
        # Dictionary to store signatures by author
        genuine_by_author = {}
        forgery_by_author = {}

        # Load genuine signatures
        for filename in os.listdir(genuine_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg','.PNG')):
                try:
                    # Parse filename, e.g., '001_02.png' -> author '001'
                    author_id = filename.split('_')[0]
                    img_path = os.path.join(genuine_dir, filename)
                    if author_id not in genuine_by_author:
                        genuine_by_author[author_id] = []
                    genuine_by_author[author_id].append(img_path)
                    self.image_cache[img_path] = Image.open(img_path).convert('L')
                except (ValueError, IndexError):
                    print(f"Skipping invalid genuine filename: {filename}")
                    continue

        # Load forged signatures
        for filename in os.listdir(forgery_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg','.PNG')):
                try:
                    # Parse filename, e.g., '0119001_01.png' -> author '001'
                    parts = filename.split('_')
                    author_id = parts[0][-3:]  # Extract last 3 digits (e.g., '001' from '0119001')
                    img_path = os.path.join(forgery_dir, filename)
                    if author_id not in forgery_by_author:
                        forgery_by_author[author_id] = []
                    forgery_by_author[author_id].append(img_path)
                    self.image_cache[img_path] = Image.open(img_path).convert('L')
                except (ValueError, IndexError):
                    print(f"Skipping invalid forgery filename: {filename}")
                    continue

        print(f"Genuine authors: {len(genuine_by_author)}, {genuine_by_author.keys()}")
        print(f"Forgery authors: {len(forgery_by_author)}, {forgery_by_author.keys()}")
        for author_id, sigs in genuine_by_author.items():
            print(f"Author {author_id}: {len(sigs)} genuine signatures")
        for author_id, sigs in forgery_by_author.items():
            print(f"Author {author_id}: {len(sigs)} forged signatures")
        print(f"Cached {len(self.image_cache)} unique images in RAM")
        # Create pairs for each author
        for author_id in genuine_by_author:
            genuine_sigs = genuine_by_author.get(author_id, [])
            forged_sigs = forgery_by_author.get(author_id, [])
            if not (forged_sigs or genuine_sigs):
                print(f"Warning: No data found for author {author_id}")

            # Genuine-Genuine pairs (label = 1)
            for i in range(len(genuine_sigs)):
                for j in range(i + 1, len(genuine_sigs)):  # Pair different genuine samples
                    self.pairs.append((genuine_sigs[i], genuine_sigs[j]))
                    self.labels.append(1)

            # Genuine-Forgery pairs (label = 0)
            indices = np.random.permutation(len(genuine_sigs))[:20]
            selected_genuines = [genuine_sigs[i] for i in indices]
            for genuine_sig in selected_genuines:
                for forged_sig in forged_sigs:
                    self.pairs.append((genuine_sig, forged_sig))
                    self.labels.append(0)

        print(f"Training Total pairs: {len(self.pairs)}, Total labels: {len(self.labels)}")
        print(f"Training Positive pairs (label=1): {sum(1 for label in self.labels if label == 1)}")
        print(f"Training Negative pairs (label=0): {sum(1 for label in self.labels if label == 0)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Get pair of image paths and label
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        # Load images
        img1 = self.image_cache[img1_path].copy()
        img2 = self.image_cache[img2_path].copy()

        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)


        return img1, img2, torch.tensor(label, dtype=torch.float32)

# Example usage
transform = transforms.Compose([
    # transforms.Resize((64, 64)),  # Resize to match FashionMNIST if needed
    # transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
    transforms.Resize((64, 64)),
    transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.92, 1.08), fill=255),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.25, fill=255),
    transforms.RandomApply(
        [transforms.GaussianBlur(3, sigma=(0.1, 0.5))],
        p=0.3
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),# <-- randomly erase patches
    # transforms.RandomErasing(p=0.05, scale=(0.02,0.05), ratio=(0.3,3.3))  
])
