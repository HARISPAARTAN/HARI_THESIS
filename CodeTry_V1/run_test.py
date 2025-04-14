import torch
from feature_extractor import ResNet50FeatureExtractor

def main():
    print("Testing ResNet50 Feature Extractor")
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    
    # Create dummy input (batch_size=2, channels=3, height=224, width=224)
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Create feature extractor
    print("Creating ResNet50 feature extractor...")
    feature_extractor = ResNet50FeatureExtractor(use_cuda=use_cuda)
    
    # Process batch
    print("Processing batch...")
    with torch.no_grad():
        results = feature_extractor.process_batch(dummy_input)
    
    # Print shapes
    print("\nOutput shapes:")
    print(f"Features shape: {results['features'].shape}")
    print(f"Patches shape: {results['patches'].shape}")
    print(f"Similarity shape: {results['similarity'].shape}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main() 