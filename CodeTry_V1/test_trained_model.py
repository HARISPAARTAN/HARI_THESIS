import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import torchvision.transforms as transforms

from feature_extractor import ResNet50FeatureExtractor
from trainable_model import TrainableFeatureExtractor, visualize_patch_similarity

def load_image(image_path):
    """Load and preprocess an image"""
    img = Image.open(image_path).convert('RGB')
    
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return preprocess(img).unsqueeze(0)  # Add batch dimension

def compute_similarity_matrix(features):
    """Compute cosine similarity matrix between patches"""
    # Extract patches: [batch_size, num_patches, channels]
    batch_size, channels, height, width = features.shape
    
    # Reshape to [batch_size, channels, height*width]
    features_flat = features.view(batch_size, channels, height * width)
    
    # Transpose to [batch_size, height*width, channels]
    patches = features_flat.permute(0, 2, 1)
    
    # Normalize features for cosine similarity
    patches_norm = F.normalize(patches[0], p=2, dim=1)  # Take first batch
    
    # Compute similarity matrix
    similarity = torch.mm(patches_norm, patches_norm.t()).cpu().numpy()
    
    return similarity

def visualize_comparison(pretrained_similarity, trained_similarity, save_path=None):
    """Visualize similarity matrices from pretrained and trained models"""
    # Calculate absolute difference
    diff = np.abs(trained_similarity - pretrained_similarity)
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot pretrained similarity
    im1 = axs[0].imshow(pretrained_similarity, cmap='viridis')
    axs[0].set_title('Pretrained Model\nPatch Similarity')
    axs[0].set_xlabel('Patch Index')
    axs[0].set_ylabel('Patch Index')
    plt.colorbar(im1, ax=axs[0])
    
    # Plot trained similarity
    im2 = axs[1].imshow(trained_similarity, cmap='viridis')
    axs[1].set_title('Trained Model\nPatch Similarity')
    axs[1].set_xlabel('Patch Index')
    axs[1].set_ylabel('Patch Index')
    plt.colorbar(im2, ax=axs[1])
    
    # Plot difference
    im3 = axs[2].imshow(diff, cmap='hot')
    axs[2].set_title('Absolute Difference')
    axs[2].set_xlabel('Patch Index')
    axs[2].set_ylabel('Patch Index')
    plt.colorbar(im3, ax=axs[2])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compute_diversity_metric(similarity_matrix):
    """Compute average off-diagonal similarity as a diversity metric"""
    # Create mask to exclude diagonal elements
    n = similarity_matrix.shape[0]
    mask = 1.0 - np.eye(n)
    
    # Compute mean of off-diagonal elements
    off_diag_mean = (similarity_matrix * mask).sum() / (n * n - n)
    
    return off_diag_mean

def main():
    parser = argparse.ArgumentParser(description='Test trained model on an image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, default='./test_output', help='Output directory')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image
    print(f"Loading image from {args.image}...")
    img_tensor = load_image(args.image).to(device)
    
    # Load pretrained feature extractor
    print("Loading pretrained ResNet50 feature extractor...")
    pretrained_model = ResNet50FeatureExtractor(use_cuda=(device == 'cuda'))
    
    # Load trained model
    print(f"Loading trained model from {args.model}...")
    trained_model = TrainableFeatureExtractor(freeze_backbone=True)
    trained_model.load_state_dict(torch.load(args.model, map_location=device))
    trained_model.to(device)
    trained_model.eval()
    
    # Extract features and compute similarity
    with torch.no_grad():
        # Pretrained model
        pretrained_features = pretrained_model.forward(img_tensor)
        pretrained_similarity = compute_similarity_matrix(pretrained_features)
        
        # Trained model
        trained_features = trained_model(img_tensor)
        trained_similarity = compute_similarity_matrix(trained_features)
    
    # Compute diversity metrics
    pretrained_diversity = compute_diversity_metric(pretrained_similarity)
    trained_diversity = compute_diversity_metric(trained_similarity)
    
    print(f"Pretrained model diversity score: {pretrained_diversity:.6f}")
    print(f"Trained model diversity score: {trained_diversity:.6f}")
    print(f"Improvement: {pretrained_diversity - trained_diversity:.6f}")
    
    # Visualize comparison
    comparison_path = os.path.join(args.output_dir, 'similarity_comparison.png')
    visualize_comparison(pretrained_similarity, trained_similarity, comparison_path)
    print(f"Similarity comparison saved to {comparison_path}")
    
    print("Done!")

if __name__ == "__main__":
    main() 