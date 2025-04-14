import os
import torch
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from feature_extractor import ResNet50FeatureExtractor
from main import (
    visualize_similarity, 
    visualize_similarity_with_values, 
    save_similarity_table, 
    save_similarity_html_table,
    create_comparison_visualization,
    visualize_patches_grid
)

def process_image(image_path, output_dir='./custom_image_test', use_cuda=True):
    """Process a custom image and generate visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print(f"Loading image from {image_path}...")
    img = Image.open(image_path).convert('RGB')
    
    # Save original unprocessed image
    original_path = os.path.join(output_dir, 'original_image.jpg')
    img.save(original_path)
    print(f"Saved original image to {original_path}")
    
    # Preprocess image for model
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    
    # Create feature extractor
    print("Creating ResNet50 feature extractor...")
    feature_extractor = ResNet50FeatureExtractor(use_cuda=use_cuda)
    
    # Process image
    print("Processing image...")
    with torch.no_grad():
        results = feature_extractor.process_batch(img_tensor)
    
    # Get results
    features = results['features'][0].cpu().numpy()
    patches = results['patches'][0].cpu().numpy()
    similarity = results['similarity'][0]
    
    # Print shapes
    print(f"Features shape: {features.shape}")  # Should be [2048, 7, 7]
    print(f"Patches shape: {patches.shape}")    # Should be [49, 2048]
    print(f"Similarity matrix shape: {similarity.shape}")  # Should be [49, 49]
    
    # Save results
    print("Saving results...")
    np.save(os.path.join(output_dir, 'features.npy'), features)
    np.save(os.path.join(output_dir, 'patches.npy'), patches)
    np.save(os.path.join(output_dir, 'similarity.npy'), similarity)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Visualize patches grid
    patches_grid_path = os.path.join(output_dir, 'patches_grid.png')
    visualize_patches_grid(img_tensor[0], patches_grid_path)
    print(f"Saved patches grid visualization to {patches_grid_path}")
    
    # Standard heatmap
    vis_path = os.path.join(output_dir, 'similarity_heatmap.png')
    visualize_similarity(similarity, vis_path)
    print(f"Saved similarity heatmap to {vis_path}")
    
    # Annotated heatmap with values
    annotated_path = os.path.join(output_dir, 'similarity_values.png')
    visualize_similarity_with_values(similarity, annotated_path)
    print(f"Saved annotated similarity values to {os.path.join(output_dir, 'similarity_values_annotated.png')}")
    
    # CSV table
    csv_path = os.path.join(output_dir, 'similarity_table.csv')
    save_similarity_table(similarity, csv_path)
    print(f"Saved CSV table to {csv_path}")
    
    # HTML table
    html_path = os.path.join(output_dir, 'similarity_table.html')
    save_similarity_html_table(similarity, html_path)
    print(f"Saved HTML table to {html_path}")
    
    # Comparison visualization
    comparison_path = os.path.join(output_dir, 'comparison.png')
    create_comparison_visualization(img_tensor[0], similarity, comparison_path)
    print(f"Saved comparison visualization to {comparison_path}")
    
    print(f"\nProcessing completed! All results saved to {os.path.abspath(output_dir)}")
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Process a custom image and visualize patch similarity')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='./custom_image_test', help='Directory to save outputs')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(f"CUDA available: {use_cuda}")
    
    # Process the image
    process_image(args.image, args.output_dir, use_cuda)

if __name__ == "__main__":
    main() 