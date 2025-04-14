import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self, use_cuda=True):
        super(ResNet50FeatureExtractor, self).__init__()
        # Load pretrained ResNet50 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the fully connected layer and average pooling
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model = self.model.to(self.device)
        
        # Freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass through the model up to the Conv5 layer
        """
        x = x.to(self.device)
        features = self.model(x)  # Shape: [batch_size, 2048, 7, 7]
        return features
    
    def extract_patches(self, features, num_patches=49):
        """
        Extract patches from the feature maps
        
        Args:
            features: Tensor of shape [batch_size, 2048, 7, 7]
            num_patches: Number of patches to extract (default: 49 for 7x7 feature map)
        
        Returns:
            patches: Tensor of shape [batch_size, num_patches, 2048]
        """
        batch_size, channels, height, width = features.shape
        
        # Reshape to [batch_size, channels, height*width]
        features_reshaped = features.view(batch_size, channels, height * width)
        
        # Transpose to [batch_size, height*width, channels]
        patches = features_reshaped.permute(0, 2, 1)
        
        # If num_patches is less than h*w, randomly select patches
        if num_patches < height * width:
            indices = torch.randperm(height * width, device=self.device)[:num_patches]
            patches = patches[:, indices, :]
        
        return patches
    
    def compute_patch_similarity(self, patches1, patches2, method='cosine'):
        """
        Compute similarity between patches of two images
        
        Args:
            patches1: Tensor of shape [batch_size, num_patches, channels]
            patches2: Tensor of shape [batch_size, num_patches, channels]
            method: Similarity method ('cosine', 'euclidean', or 'dot')
        
        Returns:
            similarity_matrix: Tensor of shape [batch_size, num_patches, num_patches]
        """
        batch_size, num_patches, channels = patches1.shape
        similarity_matrices = []
        
        for b in range(batch_size):
            p1 = patches1[b].cpu().numpy()  # [num_patches, channels]
            p2 = patches2[b].cpu().numpy()  # [num_patches, channels]
            
            if method == 'cosine':
                # Compute cosine similarity
                sim_matrix = cosine_similarity(p1, p2)
            elif method == 'euclidean':
                # Compute negative Euclidean distance (higher is more similar)
                dist_matrix = np.sqrt(((p1[:, None, :] - p2[None, :, :]) ** 2).sum(axis=2))
                sim_matrix = 1.0 / (1.0 + dist_matrix)
            elif method == 'dot':
                # Compute dot product similarity
                sim_matrix = np.dot(p1, p2.T)
            else:
                raise ValueError(f"Unknown similarity method: {method}")
            
            similarity_matrices.append(sim_matrix)
        
        return np.array(similarity_matrices)
    
    def process_batch(self, batch, extract_patches=True, compute_similarity=True):
        """
        Process a batch of images to extract features, patches, and compute similarity
        
        Args:
            batch: Tensor of shape [batch_size, channels, height, width]
            extract_patches: Whether to extract patches (default: True)
            compute_similarity: Whether to compute patch similarity (default: True)
        
        Returns:
            features: Tensor of shape [batch_size, 2048, 7, 7]
            patches: Tensor of shape [batch_size, num_patches, 2048] if extract_patches is True
            similarity: Numpy array of shape [batch_size, num_patches, num_patches] if compute_similarity is True
        """
        with torch.no_grad():
            # Extract features
            features = self.forward(batch)
            
            result = {'features': features}
            
            if extract_patches:
                # Extract patches
                patches = self.extract_patches(features)
                result['patches'] = patches
                
                if compute_similarity:
                    # Compute similarity between patches within each image
                    similarity = self.compute_patch_similarity(patches, patches)
                    result['similarity'] = similarity
            
            return result 