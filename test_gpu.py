
# %% [markdown]
# DATA
# Set device for GPU usage
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
from PIL import Image

# Set device for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PATH = './mymodel.pth'


# %%
genuine_extract_path = './content/Offline Genuine'
forgery_extract_path = './content/Offline Forgeries'
print(f"Genuine files: {os.listdir(genuine_extract_path)}")
print(f"Forgery files: {os.listdir(forgery_extract_path)}")

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
                except (ValueError, IndexError):
                    print(f"Skipping invalid genuine filename: {filename}")
                    continue

        # Load forged signatures
        for filename in os.listdir(forgery_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # Parse filename, e.g., '0119001_01.png' -> author '001'
                    parts = filename.split('_')
                    author_id = parts[0][-3:]  # Extract last 3 digits (e.g., '001' from '0119001')
                    img_path = os.path.join(forgery_dir, filename)
                    if author_id not in forgery_by_author:
                        forgery_by_author[author_id] = []
                    forgery_by_author[author_id].append(img_path)
                except (ValueError, IndexError):
                    print(f"Skipping invalid forgery filename: {filename}")
                    continue

        print(f"Genuine authors: {len(genuine_by_author)}, {genuine_by_author.keys()}")
        print(f"Forgery authors: {len(forgery_by_author)}, {forgery_by_author.keys()}")
        for author_id, sigs in genuine_by_author.items():
            print(f"Author {author_id}: {len(sigs)} genuine signatures")
        for author_id, sigs in forgery_by_author.items():
            print(f"Author {author_id}: {len(sigs)} forged signatures")

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
            for genuine_sig in genuine_sigs:
                for forged_sig in forged_sigs:
                    self.pairs.append((genuine_sig, forged_sig))
                    self.labels.append(0)

        print(f"Training Total pairs: {len(self.pairs)}, Total labels: {len(self.labels)}")
        count=0
        for pair in self.pairs:
            print(f"all: {pair} {self.labels[count]}")
            count+=1
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

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# Example usage
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to match FashionMNIST if needed
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Standard normalization
])

dataset = SignatureVerificationDataset(
    genuine_dir=genuine_extract_path,
    forgery_dir=forgery_extract_path,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)

# %% [markdown]
# 

# %%
reference_extract_path='./content/Reference(646)'
ques_extract_path = './content/Questioned(1287)'
print(f"Reference files: {os.listdir(reference_extract_path)}")
print(f"Questioned files: {os.listdir(ques_extract_path)}")

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

        print("ref",reference_by_author)
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
        print("ques",ques_genuine_by_author)

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
        print("ques",ques_forgery_by_author)

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
        count=0
        for pair in self.pairs:
            print(f"all: {pair} {self.labels[count]}")
            count+=1
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

        return img1, img2, torch.tensor(label, dtype=torch.float32)

# Example usage
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to match FashionMNIST if needed
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Standard normalization
])

testdataset = SignatureVerificationTestDataset(
    reference_extract_path,
    ques_extract_path,
    transform=transform
)

testdataloader = DataLoader(testdataset, batch_size=512, shuffle=True, num_workers=0)

# %%
import torch.nn as nn

class SiameseCNN(nn.Module):
    def __init__(self):
        super(SiameseCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128)  # Adjust based on image size
        )

    def forward(self, x1, x2):
        embedding1 = self.cnn(x1)
        embedding2 = self.cnn(x2)
        return embedding1, embedding2

# %%
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        cosine_sim = nn.functional.cosine_similarity(emb1, emb2)
        loss = label * (1 - cosine_sim) + (1 - label) * torch.clamp(cosine_sim - self.margin, min=0)
        return loss.mean()

# %%
import matplotlib.pyplot as plt
model = SiameseCNN().to(device)  # Move model to GPU
criterion = ContrastiveLoss(margin=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs=10   
epoch_losses=[]
for epoch in range(num_epochs):
    model.train()
    batch_losses=[]
    for img1, img2, labels in dataloader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)  # Move tensors to GPU
        optimizer.zero_grad()
        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, labels)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
    avg_loss = sum(batch_losses) / len(batch_losses)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
  # Simulate loss values (replace with actual losses collected during training)
plt.figure(figsize=(8, 6))
plt.plot(range(1,epoch+2), epoch_losses, marker='o', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Contrastive Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Save the model
torch.save(model.state_dict(), PATH)

# %%
# Load the model
model = SiameseCNN().to(device)  # Move model to GPU
model.load_state_dict(torch.load(PATH))

# %%
import torch
import torch.nn.functional as F

# Define a threshold for classifying pairs
THRESHOLD = 0.5  # Tune this based on your dataset (e.g., via validation set)

def evaluate_accuracy(model, testdataloader, threshold=0.5):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2, labels in testdataloader:
            # Get embeddings and cosine similarity
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)  # Move tensors to GPU
            emb1, emb2 = model(img1, img2)
            cosine_sim = F.cosine_similarity(emb1, emb2)

            # Predict based on threshold
            predictions = (cosine_sim > threshold).float()  # 1 if similar, 0 if dissimilar

            # Compare predictions with ground truth labels
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Optional: Print similarity and labels for debugging
            print(f"Cosine Similarity: {cosine_sim[:5]}")
            print(f"Ground Truth Labels: {labels[:5]}")
            print(f"Predictions: {predictions[:5]}")

    accuracy = correct / total
    return accuracy

if __name__ == '__main__':
    # Load the model
    model = SiameseCNN().to(device)  # Move model to GPU
    model.load_state_dict(torch.load(PATH))

    # Evaluate accuracy
    accuracy = evaluate_accuracy(model, testdataloader, THRESHOLD)
    print(f"Accuracy: {accuracy:.4f}")