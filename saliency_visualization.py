# -*- coding: utf-8 -*-
"""
Saliency Visualization using pytorch-grad-cam library
PRODUCTION-READY with reliable Grad-CAM++ implementation
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import cv2
import os

# Install with: pip install grad-cam
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    print("⚠️ pytorch-grad-cam not installed. Run: pip install grad-cam")
    GRADCAM_AVAILABLE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SaliencyVisualizer:
    """
    Production-ready visualization using pytorch-grad-cam library
    Falls back to custom implementation if library unavailable
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.use_library = GRADCAM_AVAILABLE
        
        if not self.use_library:
            print("⚠️ Using fallback Grad-CAM (install pytorch-grad-cam for better results)")
        
    def _get_model(self):
        """Get actual model, handling torch.compile wrapper"""
        try:
            return self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        except:
            return self.model
    
    def compute_saliency(self, img1, img2, label):
        """
        Compute vanilla gradient saliency map
        """
        if isinstance(img1, list):
            img1 = img1[0]
        if isinstance(img2, list):
            img2 = img2[0]
        
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        img1 = img1.clone().detach().to(self.device).requires_grad_(True)
        img2 = img2.clone().detach().to(self.device).requires_grad_(True)
        
        emb1 = self.model(img1)
        emb2 = self.model(img2)
        distance = F.pairwise_distance(emb1, emb2)
        
        self.model.zero_grad()
        if img1.grad is not None:
            img1.grad.zero_()
        if img2.grad is not None:
            img2.grad.zero_()
            
        distance.backward()
        
        if img1.grad is None or img2.grad is None:
            raise RuntimeError("Gradients not computed")
        
        saliency_img1 = img1.grad.abs().squeeze().cpu().numpy()
        saliency_img2 = img2.grad.abs().squeeze().cpu().numpy()
        
        return saliency_img1, saliency_img2, distance.item()
    
    def compute_gradcam_library(self, img1, img2):
        """
        Compute Grad-CAM using pytorch-grad-cam library (BEST METHOD)
        """
        if not self.use_library:
            return self.compute_gradcam_fallback(img1, img2)
        
        if isinstance(img1, list):
            img1 = img1[0]
        if isinstance(img2, list):
            img2 = img2[0]
        
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        img1 = img1.clone().detach().to(self.device)
        img2 = img2.clone().detach().to(self.device)
        
        model = self._get_model()
        target_layers = [model.block4]
        
        # Custom target for Siamese networks
        class SiameseDistanceTarget:
            def __init__(self, other_img, model_ref):
                self.other_img = other_img
                self.model_ref = model_ref
            
            def __call__(self, model_output):
                # model_output is the flattened embedding
                # We need to get the other embedding and compute distance
                with torch.no_grad():
                    other_emb = self.model_ref(self.other_img)
                
                # Reshape if needed
                if model_output.dim() == 1:
                    model_output = model_output.unsqueeze(0)
                if other_emb.dim() == 1:
                    other_emb = other_emb.unsqueeze(0)
                
                distance = F.pairwise_distance(model_output, other_emb)
                return -distance.sum()  # Negative because we want to maximize for CAM
        
        try:
            # Compute Grad-CAM for img1
            with GradCAMPlusPlus(model=self.model, target_layers=target_layers) as cam:
                targets = [SiameseDistanceTarget(img2, self.model)]
                grayscale_cam1 = cam(input_tensor=img1, targets=targets)
                cam1 = grayscale_cam1[0, :]
            
            # Compute Grad-CAM for img2
            with GradCAMPlusPlus(model=self.model, target_layers=target_layers) as cam:
                targets = [SiameseDistanceTarget(img1, self.model)]
                grayscale_cam2 = cam(input_tensor=img2, targets=targets)
                cam2 = grayscale_cam2[0, :]
            
            return cam1, cam2
            
        except Exception as e:
            print(f"Library Grad-CAM failed: {e}, using fallback")
            return self.compute_gradcam_fallback(img1, img2)
    
    def compute_gradcam_fallback(self, img1, img2):
        """
        Fallback Grad-CAM if library not available
        """
        if isinstance(img1, list):
            img1 = img1[0]
        if isinstance(img2, list):
            img2 = img2[0]
        
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        img1 = img1.clone().detach().to(self.device).requires_grad_(True)
        img2 = img2.clone().detach().to(self.device)
        
        model = self._get_model()
        
        # Forward through blocks to capture activations
        x1 = img1
        x1 = model.block1(x1)
        x1 = model.pool1(x1)
        x1 = model.block2(x1)
        x1 = model.pool2(x1)
        x1 = model.block3(x1)
        x1 = model.pool3(x1)
        x1 = model.block4(x1)
        
        block4_output = x1
        
        # Continue to embedding
        x1 = model.global_pool(x1)
        x1 = torch.flatten(x1, 1)
        x1 = model.embedding(x1)
        emb1 = F.normalize(x1, p=2, dim=1)
        
        # Get emb2
        with torch.no_grad():
            emb2 = self.model(img2)
        
        # Compute distance and backward
        distance = F.pairwise_distance(emb1, emb2)
        self.model.zero_grad()
        distance.backward(retain_graph=True)
        
        # Get gradients
        if block4_output.grad is None:
            return np.zeros((64, 64)), np.zeros((64, 64))
        
        gradients = block4_output.grad.cpu().numpy()[0]
        activations = block4_output.detach().cpu().numpy()[0]
        
        # Compute CAM
        weights = np.mean(gradients, axis=(1, 2))
        cam1 = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
        cam1 = np.maximum(cam1, 0)
        if cam1.max() > 0:
            cam1 = cam1 / cam1.max()
        cam1 = cv2.resize(cam1, (64, 64))
        
        # For img2, repeat process
        img2_req = img2.clone().detach().requires_grad_(True)
        img1_fixed = img1.clone().detach()
        
        x2 = img2_req
        x2 = model.block1(x2)
        x2 = model.pool1(x2)
        x2 = model.block2(x2)
        x2 = model.pool2(x2)
        x2 = model.block3(x2)
        x2 = model.pool3(x2)
        x2 = model.block4(x2)
        
        block4_output2 = x2
        
        x2 = model.global_pool(x2)
        x2 = torch.flatten(x2, 1)
        x2 = model.embedding(x2)
        emb2_new = F.normalize(x2, p=2, dim=1)
        
        with torch.no_grad():
            emb1_fixed = self.model(img1_fixed)
        
        distance2 = F.pairwise_distance(emb2_new, emb1_fixed)
        self.model.zero_grad()
        distance2.backward()
        
        if block4_output2.grad is not None:
            gradients2 = block4_output2.grad.cpu().numpy()[0]
            activations2 = block4_output2.detach().cpu().numpy()[0]
            
            weights2 = np.mean(gradients2, axis=(1, 2))
            cam2 = np.sum(weights2[:, np.newaxis, np.newaxis] * activations2, axis=0)
            cam2 = np.maximum(cam2, 0)
            if cam2.max() > 0:
                cam2 = cam2 / cam2.max()
            cam2 = cv2.resize(cam2, (64, 64))
        else:
            cam2 = np.zeros((64, 64))
        
        return cam1, cam2
    
    def visualize_pair(self, img1, img2, label, save_path=None, pair_idx=0):
        """
        Complete visualization with library-based Grad-CAM
        """
        if isinstance(img1, Image.Image):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            img1 = transform(img1).unsqueeze(0)
            img2 = transform(img2).unsqueeze(0)
        
        if isinstance(img1, list):
            img1 = img1[0]
        if isinstance(img2, list):
            img2 = img2[0]
        
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        # Compute visualizations
        try:
            saliency1, saliency2, distance = self.compute_saliency(img1, img2, label)
        except Exception as e:
            print(f"⚠️ Saliency failed: {e}")
            saliency1 = np.zeros((64, 64))
            saliency2 = np.zeros((64, 64))
            distance = 0.0
        
        try:
            gradcam1, gradcam2 = self.compute_gradcam_library(img1, img2)
        except Exception as e:
            print(f"⚠️ Grad-CAM failed: {e}")
            import traceback
            traceback.print_exc()
            gradcam1 = np.zeros((64, 64))
            gradcam2 = np.zeros((64, 64))
        
        # Get original images
        img1_np = img1.squeeze().cpu().detach().numpy()
        img2_np = img2.squeeze().cpu().detach().numpy()
        img1_np = (img1_np * 0.5) + 0.5
        img2_np = (img2_np * 0.5) + 0.5
        img1_np = np.clip(img1_np, 0, 1)
        img2_np = np.clip(img2_np, 0, 1)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        label_text = "GENUINE PAIR" if label == 1 else "FORGED PAIR"
        method = "Grad-CAM++" if self.use_library else "Grad-CAM (fallback)"
        fig.suptitle(f'{label_text} - Distance: {distance:.4f} [{method}]', 
                     fontsize=16, fontweight='bold')
        
        # Row 1: Original Images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img1_np, cmap='gray')
        ax1.set_title('Reference Signature')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img2_np, cmap='gray')
        ax2.set_title('Query Signature')
        ax2.axis('off')
        
        ax_info = fig.add_subplot(gs[0, 2:])
        ax_info.axis('off')
        info_text = f"""
        Model Prediction:
        Distance: {distance:.4f}
        Ground Truth: {"Genuine" if label == 1 else "Forged"}
        
        Low distance → Similar (Genuine)
        High distance → Different (Forged)
        """
        ax_info.text(0.1, 0.5, info_text, fontsize=12, 
                     verticalalignment='center', family='monospace')
        
        # Row 2: Saliency Maps
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(saliency1, cmap='hot')
        ax3.set_title('Saliency Map - Ref')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        ax4 = fig.add_subplot(gs[1, 1])
        im4 = ax4.imshow(saliency2, cmap='hot')
        ax4.set_title('Saliency Map - Query')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.imshow(img1_np, cmap='gray', alpha=0.7)
        ax5.imshow(saliency1, cmap='hot', alpha=0.5)
        ax5.set_title('Saliency Overlay - Ref')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 3])
        ax6.imshow(img2_np, cmap='gray', alpha=0.7)
        ax6.imshow(saliency2, cmap='hot', alpha=0.5)
        ax6.set_title('Saliency Overlay - Query')
        ax6.axis('off')
        
        # Row 3: Grad-CAM
        ax7 = fig.add_subplot(gs[2, 0])
        im7 = ax7.imshow(gradcam1, cmap='jet')
        ax7.set_title(f'{method} - Ref')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        ax8 = fig.add_subplot(gs[2, 1])
        im8 = ax8.imshow(gradcam2, cmap='jet')
        ax8.set_title(f'{method} - Query')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046)
        
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.imshow(img1_np, cmap='gray')
        ax9.imshow(gradcam1, cmap='jet', alpha=0.4)
        ax9.set_title('Grad-CAM Overlay - Ref')
        ax9.axis('off')
        
        ax10 = fig.add_subplot(gs[2, 3])
        ax10.imshow(img2_np, cmap='gray')
        ax10.imshow(gradcam2, cmap='jet', alpha=0.4)
        ax10.set_title('Grad-CAM Overlay - Query')
        ax10.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = f"{save_path}/pair_{pair_idx}_{label_text.lower()}_dist{distance:.3f}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
        plt.close()


def analyze_model_focus(model, dataloader, device, num_samples=10, save_dir='saliency_outputs'):
    """Analyze model focus"""
    visualizer = SaliencyVisualizer(model, device)
    os.makedirs(save_dir, exist_ok=True)
    
    genuine_pairs = []
    forged_pairs = []
    
    print(f"Collecting {num_samples} genuine and {num_samples} forged pairs...")
    
    with torch.no_grad():
        for img1_array, img2, labels in dataloader:
            if isinstance(img1_array, list):
                img1 = img1_array[0]
            else:
                img1 = img1_array
            
            for i in range(len(labels)):
                if labels[i] == 1 and len(genuine_pairs) < num_samples:
                    genuine_pairs.append((img1[i:i+1].clone(), img2[i:i+1].clone(), 1))
                elif labels[i] == 0 and len(forged_pairs) < num_samples:
                    forged_pairs.append((img1[i:i+1].clone(), img2[i:i+1].clone(), 0))
                
                if len(genuine_pairs) >= num_samples and len(forged_pairs) >= num_samples:
                    break
            
            if len(genuine_pairs) >= num_samples and len(forged_pairs) >= num_samples:
                break
    
    print(f"✓ Collected {len(genuine_pairs)} genuine and {len(forged_pairs)} forged pairs")
    
    print("\nVisualizing pairs...")
    for idx, (img1, img2, label) in enumerate(genuine_pairs + forged_pairs):
        print(f"Processing pair {idx+1}/{len(genuine_pairs)+len(forged_pairs)}...")
        visualizer.visualize_pair(img1, img2, label, save_path=save_dir, pair_idx=idx)
    
    print(f"\n✓ All visualizations saved to: {save_dir}")


if __name__ == "__main__":
    print("Install grad-cam library with: pip install grad-cam")
    print("Then import and use SaliencyVisualizer in your notebook")