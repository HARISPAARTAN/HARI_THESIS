import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import einops

class TrainableFeatureExtractor(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(TrainableFeatureExtractor, self).__init__()
        # Load pretrained ResNet50 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the fully connected layer and average pooling
        self.backbone = nn.Sequential(*list(self.model.children())[:-2])
        
        # Add a trainable projection layer
        self.projection = nn.Conv2d(2048, 512, kernel_size=1)
        
        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # Get features from backbone (ResNet50 up to Conv5)
        features = self.backbone(x)  # Shape: [batch_size, 2048, 7, 7]
        
        # Apply projection layer
        projected = self.projection(features)  # Shape: [batch_size, 512, 7, 7]
        
        return projected
    
    def extract_patches(self, features):
        """
        Extract patches from the feature maps
        Args:
            features: Tensor of shape [batch_size, channels, 7, 7]
        Returns:
            patches: Tensor of shape [batch_size, 49, channels]
        """
        batch_size, channels, height, width = features.shape
        
        # Reshape to [batch_size, channels, height*width]
        features_reshaped = features.view(batch_size, channels, height * width)
        
        # Transpose to [batch_size, height*width, channels]
        patches = features_reshaped.permute(0, 2, 1)
        
        return patches

def diversity_loss(features):
    """
    Compute diversity loss to encourage different patches in the feature map
    
    Args:
        features: Tensor of shape [batch_size, channels, height, width]
    
    Returns:
        loss: Scalar tensor representing the diversity loss
    """
    batch_size, channels, height, width = features.shape
    
    # Reshape to [batch_size, channels, height*width]
    x = einops.rearrange(features, 'b c h w -> b c (h w)')
    
    # Normalize features for cosine similarity
    x_norm = F.normalize(x, p=2, dim=1)
    
    # Compute similarity matrix: [batch_size, height*width, height*width]
    similarity = torch.bmm(x_norm.transpose(1, 2), x_norm)
    
    # We want off-diagonal elements to be minimized (dissimilar patches)
    # Create a mask to zero out the diagonal
    mask = 1.0 - torch.eye(height * width, device=features.device)
    masked_similarity = similarity * mask
    
    # Compute mean similarity loss (we want to minimize this)
    diversity_loss = masked_similarity.abs().mean()
    
    return diversity_loss

def train_model(model, dataloader, optimizer, epochs=10, device='cuda'):
    """
    Train the model to learn diverse patch representations
    
    Args:
        model: TrainableFeatureExtractor model
        dataloader: DataLoader providing batches of images
        optimizer: Optimizer for updating model parameters
        epochs: Number of training epochs
        device: Device to perform computations on
    """
    model.to(device)
    model.train()
    
    epoch_losses = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            features = model(images)
            
            # Compute loss
            loss = diversity_loss(features)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        # Compute average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return epoch_losses

def visualize_training(losses, save_path=None):
    """Visualize training losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_patch_similarity(model, image_tensor, save_path=None):
    """
    Visualize the patch similarity matrix before and after training
    
    Args:
        model: Trained model
        image_tensor: Input image tensor [1, 3, height, width]
        save_path: Path to save the visualization
    """
    model.eval()
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        features = model(image_tensor)
        patches = model.extract_patches(features)[0]  # Take first batch
        
        # Compute cosine similarity
        patches_norm = F.normalize(patches, p=2, dim=1)
        similarity = torch.mm(patches_norm, patches_norm.t()).cpu().numpy()
    
    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity, cmap='viridis')
    plt.colorbar(label='Similarity')
    plt.title('Patch Similarity Matrix (After Training)')
    plt.xlabel('Patch Index')
    plt.ylabel('Patch Index')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    from data_loader import download_tiny_imagenet, get_data_loaders
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a diverse patch model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to store dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./trained_model', help='Output directory')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download dataset
    dataset_path = download_tiny_imagenet(args.data_dir)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(dataset_path, batch_size=args.batch_size)
    
    # Create model
    model = TrainableFeatureExtractor(freeze_backbone=True)
    
    # Create optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    # Train model
    print("Training model...")
    losses = train_model(model, train_loader, optimizer, epochs=args.epochs, device=device)
    
    # Visualize training progress
    visualize_training(losses, save_path=os.path.join(args.output_dir, 'training_loss.png'))
    
    # Save model
    model_path = os.path.join(args.output_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Visualize patch similarity on a validation image
    print("Visualizing patch similarity...")
    val_iter = iter(val_loader)
    val_images, _ = next(val_iter)
    visualize_patch_similarity(model, val_images[0:1], 
                              save_path=os.path.join(args.output_dir, 'patch_similarity.png'))
    
    print("Done!") 