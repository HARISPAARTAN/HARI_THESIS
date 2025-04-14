import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import torchvision.transforms as transforms

from trainable_model import TrainableFeatureExtractor

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

def compute_similarity_matrix(model, img_tensor, device):
    """Compute cosine similarity matrix for image patches"""
    img_tensor = img_tensor.to(device)
    model.eval()
    
    with torch.no_grad():
        # Extract features
        features = model(img_tensor)
        
        # Extract patches
        batch_size, channels, height, width = features.shape
        features_flat = features.view(batch_size, channels, height * width)
        patches = features_flat.permute(0, 2, 1)[0]  # Take first batch
        
        # Normalize patches
        patches_norm = F.normalize(patches, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(patches_norm, patches_norm.t()).cpu().numpy()
    
    return similarity

def visualize_similarity(similarity_matrix, save_path=None):
    """Visualize similarity matrix"""
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Similarity')
    plt.title('Patch Similarity Matrix')
    plt.xlabel('Patch Index')
    plt.ylabel('Patch Index')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compute_diversity_metric(similarity_matrix):
    """Compute diversity metric (average off-diagonal similarity)"""
    n = similarity_matrix.shape[0]
    mask = 1.0 - np.eye(n)
    off_diag_mean = (similarity_matrix * mask).sum() / (n * n - n)
    return off_diag_mean

def main():
    parser = argparse.ArgumentParser(description='Test trained model')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image
    print(f"Loading image from {args.image}...")
    img_tensor = load_image(args.image)
    
    # Load trained model
    print(f"Loading trained model from {args.model}...")
    model = TrainableFeatureExtractor(freeze_backbone=True)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity = compute_similarity_matrix(model, img_tensor, device)
    
    # Compute diversity metric
    diversity = compute_diversity_metric(similarity)
    print(f"Diversity metric: {diversity:.6f}")
    
    # Visualize similarity matrix
    vis_path = os.path.join(args.output_dir, 'similarity_matrix.png')
    visualize_similarity(similarity, vis_path)
    print(f"Similarity matrix visualization saved to {vis_path}")
    
    # Save raw similarity matrix
    np.save(os.path.join(args.output_dir, 'similarity_matrix.npy'), similarity)
    print(f"Raw similarity matrix saved to {os.path.join(args.output_dir, 'similarity_matrix.npy')}")
    
    print("Done!")

if __name__ == "__main__":
    main() 