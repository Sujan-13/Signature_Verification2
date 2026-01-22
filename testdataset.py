import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

reference_extract_path='./content/Reference(646)'
ques_extract_path = './content/Questioned(1287)'
class SignatureVerificationTestDataset(Dataset):
    def __init__(self, reference_dir, ques_dir, transform=None):
        """
        Args:
            reference_dir (str): Path to offline_genuine folder (e.g., './data/offline_genuine').
            ques_dir (str): Path to offline_forgeries folder (e.g., './data/offline_forgeries').
            transform (callable, optional): Transformations to apply to images.
        """
        self.reference_dir = reference_dir
        self.ques_dir = ques_dir
        self.transform = transform
        self.pairs = []
        self.labels = []

        # Dictionary to store signatures by author
        reference_by_author = {}
        ques_genuine_by_author = {}
        ques_forgery_by_author = {}

        # Load reference signatures
        for author_id in os.listdir(reference_dir):
            for filename in os.listdir(os.path.join(reference_dir,author_id)):
                if filename.endswith(('.png', '.jpg', '.jpeg','.PNG')):
                    try:
                        # Parse filename, e.g., '001_02.png' -> author '001'
                        img_path = os.path.join(reference_dir,author_id, filename)
                        if author_id not in reference_by_author:
                            reference_by_author[author_id] = []
                        reference_by_author[author_id].append(img_path)
                    except (ValueError, IndexError):
                        print(f"Skipping invalid genuine filename: {filename}")
                        continue
        # Load questioned genuine signatures
        for author_id in os.listdir(ques_dir):
            for filename in os.listdir(os.path.join(ques_dir,author_id)):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Parse filename, e.g., '001_02.png' -> author '001'
                        img_path = os.path.join(ques_dir,author_id, filename)
                        if author_id not in ques_genuine_by_author:
                            ques_genuine_by_author[author_id] = []
                        ques_genuine_by_author[author_id].append(img_path)
                    except (ValueError, IndexError):
                        print(f"Skipping invalid genuine filename: {filename}")
                        continue

        # Load questioned forgery signatures
        for author_id in os.listdir(ques_dir):
            for filename in os.listdir(os.path.join(ques_dir,author_id)):
                if filename.endswith(('.PNG')):
                    try:
                        # Parse filename, e.g., '001_02.png' -> author '001'
                        img_path = os.path.join(ques_dir,author_id, filename)
                        if author_id not in ques_forgery_by_author:
                            ques_forgery_by_author[author_id] = []
                        ques_forgery_by_author[author_id].append(img_path)
                    except (ValueError, IndexError):
                        print(f"Skipping invalid genuine filename: {filename}")
                        continue

        # Create pairs for each author
        for author_id in reference_by_author:
            reference_sigs = reference_by_author.get(author_id, [])
            ques_genuine_sigs= ques_genuine_by_author.get(author_id,[])
            ques_forged_sigs = ques_forgery_by_author.get(author_id, [])
            if not (reference_sigs or ques_forgery_by_author or ques_genuine_by_author):
                print(f"Warning: No data found for author {author_id}")

            # Reference-Genuine pairs (label = 1)
            for i in range(len(reference_sigs)):
                for j in range(len(ques_genuine_sigs)):  # Pair different genuine samples
                    self.pairs.append((reference_sigs[i], ques_genuine_sigs[j]))
                    self.labels.append(1)

            # Reference-Forgery pairs (label = 0)
            for i in range(len(reference_sigs)):
                for j in range(len(ques_forged_sigs)):  # Pair different genuine samples
                    self.pairs.append((reference_sigs[i], ques_forged_sigs[j]))
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
        img1 = Image.open(img1_path).convert('L')  # Grayscale, like FashionMNIST
        img2 = Image.open(img2_path).convert('L')

        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(label, dtype=torch.float32)

        return img1, img2, label

# Example usage
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to match FashionMNIST if needed
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Standard normalization
])