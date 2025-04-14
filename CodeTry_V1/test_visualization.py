import os
import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from feature_extractor import ResNet50FeatureExtractor
from main import (
    visualize_similarity, 
    visualize_similarity_with_values, 
    save_similarity_table, 
    save_similarity_html_table,
    save_image,
    create_comparison_visualization
)

def load_sample_image(url=None):
    """Load a sample image either from URL or use a default one"""
    try:
        if url is None:
            # Use a reliable image URL
            url = "https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png"
        
        print(f"Trying to download image from {url}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        try:
            img = Image.open(BytesIO(response.content)).convert('RGB')
            print("Successfully downloaded and opened image from URL")
        except Exception as e:
            print(f"Error opening image from URL: {e}")
            raise
    
    except Exception as e:
        print(f"Failed to load image from URL: {e}")
        print("Using built-in sample image from torchvision instead...")
        
        # Create a simple gradient image as fallback
        img = Image.new('RGB', (224, 224), color=(255, 255, 255))
        
        # Draw a simple pattern on the image
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        for i in range(0, 224, 20):
            color = (i % 255, (i * 2) % 255, (i * 3) % 255)
            draw.rectangle([i, i, 224-i, 224-i], outline=color)
        
        print("Created a fallback sample image")
    
    # Apply preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = preprocess(img)
    # Add batch dimension
    return img_tensor.unsqueeze(0), img

def main():
    print("Testing visualizations for patch similarity")
    
    # Create output directory
    output_dir = './visualization_test'
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    
    # Load a real image instead of random noise
    print("Loading sample image...")
    sample_input, original_img = load_sample_image()
    
    # Save the original unprocessed image for reference
    original_img_path = os.path.join(output_dir, 'original_unprocessed.jpg')
    original_img.save(original_img_path)
    print(f"Saved original unprocessed image to {original_img_path}")
    
    # Create feature extractor
    print("Creating ResNet50 feature extractor...")
    feature_extractor = ResNet50FeatureExtractor(use_cuda=use_cuda)
    
    # Process batch
    print("Processing batch...")
    with torch.no_grad():
        results = feature_extractor.process_batch(sample_input)
    
    # Get similarity matrix
    similarity = results['similarity'][0]  # First image in batch
    
    # Print shape
    print(f"Similarity matrix shape: {similarity.shape}")
    
    # Save various visualizations
    print("\nGenerating visualizations...")
    
    # Save processed image
    img_path = os.path.join(output_dir, 'processed_image.png')
    save_image(sample_input[0], img_path)
    print(f"Saved processed image to {img_path}")
    
    # Standard heatmap
    vis_path = os.path.join(output_dir, 'similarity_heatmap.png')
    visualize_similarity(similarity, vis_path)
    print(f"Saved standard heatmap to {vis_path}")
    
    # Annotated heatmap
    visualize_similarity_with_values(similarity, vis_path)
    annotated_path = os.path.join(output_dir, 'similarity_heatmap_annotated.png')
    print(f"Saved annotated heatmap to {annotated_path}")
    
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
    create_comparison_visualization(sample_input[0], similarity, comparison_path)
    print(f"Saved comparison visualization to {comparison_path}")
    
    print("\nTest completed successfully!")
    print(f"Check the output directory: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main() 