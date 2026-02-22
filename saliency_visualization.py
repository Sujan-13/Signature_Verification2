# -*- coding: utf-8 -*-
"""
Saliency Map & Grad-CAM Visualization for Siamese CNN
COMPLETE FIXED VERSION with proper Grad-CAM for Siamese networks
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image
import cv2
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SaliencyVisualizer:
    """
    Visualizes what the Siamese CNN is learning using:
    1. Vanilla Gradient (Saliency Maps)
    2. Grad-CAM (Class Activation Maps) - FIXED for Siamese networks
    3. Feature Map Activations
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Hook storage
        self.gradients = None
        self.activations = None
        
    def save_gradient(self, grad):
        """Hook to save gradients"""
        self.gradients = grad
        
    def save_activation(self, module, input, output):
        """Hook to save activations"""
        self.activations = output.detach()
    
    def _get_model(self):
        """Get the actual model, handling torch.compile wrapper"""
        try:
            return self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        except:
            return self.model
    
    def compute_saliency(self, img1, img2, label):
        """
        Compute vanilla gradient saliency map
        
        Args:
            img1, img2: Input image tensors (1, 1, 64, 64) or list
            label: Ground truth label (1=genuine, 0=forged)
            
        Returns:
            saliency_img1, saliency_img2, distance: Saliency maps and distance
        """
        # Handle list input
        if isinstance(img1, list):
            img1 = img1[0]
        if isinstance(img2, list):
            img2 = img2[0]
        
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        # Clone and detach first, then move to device and enable gradients
        img1 = img1.clone().detach().to(self.device).requires_grad_(True)
        img2 = img2.clone().detach().to(self.device).requires_grad_(True)
        
        # Forward pass
        emb1 = self.model(img1)
        emb2 = self.model(img2)
        
        # Compute distance
        distance = F.pairwise_distance(emb1, emb2)
        
        # Backward pass
        self.model.zero_grad()
        if img1.grad is not None:
            img1.grad.zero_()
        if img2.grad is not None:
            img2.grad.zero_()
            
        distance.backward()
        
        # Get gradients - check if they exist
        if img1.grad is None or img2.grad is None:
            raise RuntimeError("Gradients not computed. Model might not support gradient computation.")
        
        saliency_img1 = img1.grad.abs().squeeze().cpu().numpy()
        saliency_img2 = img2.grad.abs().squeeze().cpu().numpy()
        
        return saliency_img1, saliency_img2, distance.item()
    
    def compute_gradcam_siamese(self, img1, img2, target_block='block4'):
        """
        Compute Grad-CAM for Siamese network using distance-based gradients
        
        This is FIXED for Siamese networks - uses pairwise distance as signal
        instead of embedding norm, giving meaningful discriminative gradients.
        
        Args:
            img1: Primary image to compute CAM for
            img2: Comparison image (provides discrimination context)
            target_block: Which block to visualize ('block1', 'block2', 'block3', 'block4')
            
        Returns:
            cam: Class activation map showing what img1 features matter for distance
        """
        # Handle list input
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
        
        # Get the actual model
        model = self._get_model()
        
        # Get target block
        if not hasattr(model, target_block):
            raise ValueError(f"Model doesn't have {target_block}")
        
        target_layer = getattr(model, target_block)
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(self.save_activation)
        backward_handle = target_layer.register_full_backward_hook(
            lambda module, grad_in, grad_out: self.save_gradient(grad_out[0])
        )
        
        try:
            # Forward pass for BOTH images
            emb1 = self.model(img1)
            emb2 = self.model(img2)
            
            # Compute distance - this is the task-relevant signal!
            distance = F.pairwise_distance(emb1, emb2)
            
            # Backward through img1's path
            self.model.zero_grad()
            if img1.grad is not None:
                img1.grad.zero_()
            
            distance.backward()
            
            # Compute CAM
            if self.gradients is None or self.activations is None:
                raise ValueError("Gradients or activations not captured")
            
            gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
            activations = self.activations.cpu().numpy()[0]  # (C, H, W)
            
            # Weight channels by gradients
            weights = np.mean(gradients, axis=(1, 2))  # (C,)
            cam = np.sum(weights[:, np.newaxis, np.newaxis] * activations, axis=0)
            
            # ReLU and normalize
            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Resize to input size
            cam = cv2.resize(cam, (64, 64))
            
        finally:
            forward_handle.remove()
            backward_handle.remove()
            self.gradients = None
            self.activations = None
        
        return cam
    
    def get_feature_maps(self, img, blocks_to_visualize=None):
        """
        Extract feature maps from intermediate blocks
        
        Args:
            img: Input tensor (1, 1, 64, 64) or list
            blocks_to_visualize: Which blocks to visualize (default: all)
            
        Returns:
            feature_maps: Dict of {block_name: activations}
        """
        if blocks_to_visualize is None:
            blocks_to_visualize = ['block1', 'block2', 'block3', 'block4']
        
        # Handle list input
        if isinstance(img, list):
            img = img[0]
        
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        img = img.to(self.device)
        feature_maps = {}
        
        # Get the actual model
        model = self._get_model()
        
        with torch.no_grad():
            x = img
            
            # Block 1
            if 'block1' in blocks_to_visualize:
                x = model.block1(x)
                feature_maps['block1'] = x.detach().cpu()
            else:
                x = model.block1(x)
            x = model.pool1(x)
            
            # Block 2
            if 'block2' in blocks_to_visualize:
                x = model.block2(x)
                feature_maps['block2'] = x.detach().cpu()
            else:
                x = model.block2(x)
            x = model.pool2(x)
            
            # Block 3
            if 'block3' in blocks_to_visualize:
                x = model.block3(x)
                feature_maps['block3'] = x.detach().cpu()
            else:
                x = model.block3(x)
            x = model.pool3(x)
            
            # Block 4
            if 'block4' in blocks_to_visualize:
                x = model.block4(x)
                feature_maps['block4'] = x.detach().cpu()
        
        return feature_maps
    
    def visualize_pair(self, img1, img2, label, save_path=None, pair_idx=0):
        """
        Complete visualization for a signature pair
        
        Args:
            img1, img2: Image tensors (1, 1, 64, 64) or list or PIL Images
            label: Ground truth (1=genuine, 0=forged)
            save_path: Where to save the figure
            pair_idx: Index for filename
        """
        # Convert PIL to tensor if needed
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
        
        # Handle list input
        if isinstance(img1, list):
            img1 = img1[0]
        if isinstance(img2, list):
            img2 = img2[0]
        
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        # Compute all visualizations
        try:
            saliency1, saliency2, distance = self.compute_saliency(img1, img2, label)
        except Exception as e:
            print(f"⚠️ Saliency computation failed: {e}")
            saliency1 = np.zeros((64, 64))
            saliency2 = np.zeros((64, 64))
            distance = 0.0
        
        try:
            # Use FIXED Grad-CAM for Siamese networks
            gradcam1 = self.compute_gradcam_siamese(img1, img2, target_block='block4')
            gradcam2 = self.compute_gradcam_siamese(img2, img1, target_block='block4')
        except Exception as e:
            print(f"⚠️ Grad-CAM failed: {e}")
            import traceback
            traceback.print_exc()
            gradcam1 = np.zeros((64, 64))
            gradcam2 = np.zeros((64, 64))
        
        # Get original images for visualization
        img1_np = img1.squeeze().cpu().detach().numpy()
        img2_np = img2.squeeze().cpu().detach().numpy()    
        # Denormalize for display
        img1_np = (img1_np * 0.5) + 0.5
        img2_np = (img2_np * 0.5) + 0.5
        img1_np = np.clip(img1_np, 0, 1)
        img2_np = np.clip(img2_np, 0, 1)
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Title
        label_text = "GENUINE PAIR" if label == 1 else "FORGED PAIR"
        fig.suptitle(f'{label_text} - Distance: {distance:.4f}', 
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
        
        # Distance info
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
        
        # Row 2: Saliency Maps (What pixels matter most?)
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
        
        # Overlay saliency on original
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
        
        # Row 3: Grad-CAM (Where is the network looking?)
        ax7 = fig.add_subplot(gs[2, 0])
        im7 = ax7.imshow(gradcam1, cmap='jet')
        ax7.set_title('Grad-CAM - Ref')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        ax8 = fig.add_subplot(gs[2, 1])
        im8 = ax8.imshow(gradcam2, cmap='jet')
        ax8.set_title('Grad-CAM - Query')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046)
        
        # Overlay Grad-CAM on original
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
    
    def visualize_feature_maps(self, img, save_path=None, img_idx=0, max_channels=8):
        """
        Visualize intermediate feature map activations
        
        Args:
            img: Input tensor (1, 1, 64, 64) or list
            save_path: Where to save the figure
            img_idx: Index for filename
            max_channels: Number of channels to show per block
        """
        # Handle list input
        if isinstance(img, list):
            img = img[0]
        
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        feature_maps = self.get_feature_maps(img)
        
        n_blocks = len(feature_maps)
        fig, axes = plt.subplots(n_blocks, max_channels, 
                                figsize=(max_channels * 2, n_blocks * 2))
        
        if n_blocks == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Feature Map Activations (First {max_channels} Channels per Block)', 
                    fontsize=14)
        
        for block_idx, (block_name, fmap) in enumerate(feature_maps.items()):
            fmap_np = fmap.squeeze().numpy()  # (C, H, W)
            n_channels = min(max_channels, fmap_np.shape[0])
            
            for ch_idx in range(n_channels):
                ax = axes[block_idx, ch_idx]
                ax.imshow(fmap_np[ch_idx], cmap='viridis')
                ax.axis('off')
                
                if ch_idx == 0:
                    shape_str = f"{fmap_np.shape[1]}×{fmap_np.shape[2]}"
                    ax.set_ylabel(f'{block_name}\n{fmap_np.shape[0]}ch\n{shape_str}', 
                                 fontsize=9)
            
            # Hide unused channels
            for ch_idx in range(n_channels, max_channels):
                axes[block_idx, ch_idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = f"{save_path}/feature_maps_{img_idx}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        
        plt.show()
        plt.close()


def analyze_model_focus(model, dataloader, device, num_samples=10, save_dir='saliency_outputs'):
    """
    Analyze what the model is learning by visualizing multiple pairs
    
    Args:
        model: Trained Siamese CNN
        dataloader: Test/validation dataloader
        device: cuda/cpu
        num_samples: How many pairs to visualize
        save_dir: Where to save visualizations
    """
    visualizer = SaliencyVisualizer(model, device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get genuine and forged pairs
    genuine_pairs = []
    forged_pairs = []
    
    print(f"Collecting {num_samples} genuine and {num_samples} forged pairs...")
    
    with torch.no_grad():
        for img1_array, img2, labels in dataloader:
            # Handle multiple reference images
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
    
    # Visualize genuine pairs
    print("\n" + "="*60)
    print("Visualizing GENUINE pairs (model should focus on stroke similarities)")
    print("="*60)
    for idx, (img1, img2, label) in enumerate(genuine_pairs):
        print(f"Processing genuine pair {idx+1}/{len(genuine_pairs)}...")
        visualizer.visualize_pair(img1, img2, label, save_path=save_dir, pair_idx=idx)
    
    # Visualize forged pairs
    print("\n" + "="*60)
    print("Visualizing FORGED pairs (model should detect differences)")
    print("="*60)
    for idx, (img1, img2, label) in enumerate(forged_pairs):
        print(f"Processing forged pair {idx+1}/{len(forged_pairs)}...")
        visualizer.visualize_pair(img1, img2, label, save_path=save_dir, pair_idx=idx+num_samples)
    
    print(f"\n✓ All visualizations saved to: {save_dir}")
    
    # Analysis summary
    print("\n" + "="*60)
    print("HOW TO INTERPRET THE VISUALIZATIONS:")
    print("="*60)
    print("""
    1. SALIENCY MAPS (Hot colors):
       - Show which pixels have the most influence on the model's decision
       - For genuine pairs: Should highlight stroke patterns that match
       - For forged pairs: Should highlight areas where signatures differ
    
    2. GRAD-CAM (Jet colors):
       - Shows which regions the CNN's last conv layer is focusing on
       - Red/yellow = high activation (model is "looking" here)
       - Blue = low activation (model ignores this area)
       - Good model: Focuses on signature strokes, not background
    
    3. WHAT TO LOOK FOR:
       ✓ Model focuses on actual signature strokes (not edges of image)
       ✓ Saliency highlights distinctive writing patterns
       ✓ Similar focus areas for genuine pairs
       ✓ Different focus areas for forged pairs
       
       ✗ Model focuses on image corners/edges (it's learning artifacts)
       ✗ Random scattered activations (model is confused)
       ✗ Same focus regardless of genuine/forged (model isn't discriminating)
    """)


if __name__ == "__main__":
    print("Import this module and use SaliencyVisualizer or analyze_model_focus() in your notebook")