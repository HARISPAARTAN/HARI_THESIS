import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torchvision.transforms as transforms
from PIL import Image

from data_loader import download_tiny_imagenet, get_data_loaders
from feature_extractor import ResNet50FeatureExtractor

def check_gpu():
    """Check if GPU is available and print device information"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {device_name}")
        # Print CUDA version
        cuda_version = torch.version.cuda
        print(f"CUDA Version: {cuda_version}")
        return True
    else:
        print("No GPU available, using CPU instead.")
        return False

def visualize_similarity(similarity_matrix, save_path=None):
    """Visualize the similarity matrix"""
    plt.figure(figsize=(12, 10))
    im = plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Similarity')
    plt.title('Full Patch Similarity Matrix (49×49)')
    
    # Add labels for x and y axes
    plt.xlabel('Patch Index')
    plt.ylabel('Patch Index')
    
    # Add grid lines to separate patches
    plt.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_similarity_with_values(similarity_matrix, save_path=None):
    """Visualize the similarity matrix with numerical values"""
    # Get a 7x7 section of the matrix (using first 7x7 grid) if it's larger
    height, width = similarity_matrix.shape
    if height > 7 and width > 7:
        sub_matrix = similarity_matrix[:7, :7]
    else:
        sub_matrix = similarity_matrix
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sub_matrix, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Similarity')
    ax.set_title('Patch Similarity Matrix with Values (Showing 7×7 subset of 49×49 matrix)')
    
    # Add labels for x and y axes
    ax.set_xlabel('Patch Index')
    ax.set_ylabel('Patch Index')
    
    # Add text annotations
    for i in range(sub_matrix.shape[0]):
        for j in range(sub_matrix.shape[1]):
            value = sub_matrix[i, j]
            text_color = 'white' if value < 0.7 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color)
    
    plt.tight_layout()
    
    if save_path:
        # Create a separate file for the annotated version
        base, ext = os.path.splitext(save_path)
        annotated_path = f"{base}_annotated{ext}"
        os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
        plt.savefig(annotated_path)
        plt.close()
    else:
        plt.show()

def save_similarity_table(similarity_matrix, save_path):
    """Save similarity matrix as a CSV table"""
    # Get a 7x7 section of the matrix (using first 7x7 grid) if it's larger
    height, width = similarity_matrix.shape
    if height > 7 and width > 7:
        sub_matrix = similarity_matrix[:7, :7]
    else:
        sub_matrix = similarity_matrix
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as CSV
    np.savetxt(save_path, sub_matrix, delimiter=',', fmt='%.4f')

def save_similarity_html_table(similarity_matrix, save_path):
    """Save similarity matrix as an HTML table with color-coded cells"""
    # Get a 7x7 section of the matrix (using first 7x7 grid) if it's larger
    height, width = similarity_matrix.shape
    if height > 7 and width > 7:
        sub_matrix = similarity_matrix[:7, :7]
    else:
        sub_matrix = similarity_matrix
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Generate HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Patch Similarity Matrix</title>
        <style>
            table {
                border-collapse: collapse;
                font-family: Arial, sans-serif;
            }
            th, td {
                border: 1px solid #dddddd;
                text-align: center;
                padding: 8px;
                width: 60px;
            }
            th {
                background-color: #f2f2f2;
            }
            .high-similarity {
                background-color: #ffeb3b;
                font-weight: bold;
            }
            .med-similarity {
                background-color: #a5d6a7;
            }
            .low-similarity {
                background-color: #90caf9;
            }
        </style>
    </head>
    <body>
        <h2>Patch Similarity Matrix (7x7 section)</h2>
        <table>
    """
    
    # Add table headers (optional)
    html_content += "<tr><th></th>"
    for j in range(sub_matrix.shape[1]):
        html_content += f"<th>P{j}</th>"
    html_content += "</tr>"
    
    # Add table data with color coding
    for i in range(sub_matrix.shape[0]):
        html_content += f"<tr><th>P{i}</th>"
        for j in range(sub_matrix.shape[1]):
            value = sub_matrix[i, j]
            
            # Determine cell class based on similarity value
            if value > 0.9:
                cell_class = "high-similarity"
            elif value > 0.7:
                cell_class = "med-similarity"
            else:
                cell_class = "low-similarity"
                
            html_content += f'<td class="{cell_class}">{value:.4f}</td>'
        html_content += "</tr>"
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(save_path, 'w') as f:
        f.write(html_content)

def save_image(tensor_img, save_path):
    """Save a tensor image to disk"""
    # Convert tensor to PIL Image
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
        transforms.ToPILImage()
    ])
    
    img = inv_normalize(tensor_img.cpu())
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save image
    img.save(save_path)

def create_comparison_visualization(img_tensor, similarity_matrix, save_path):
    """Create a side-by-side comparison of original image and similarity matrix"""
    # Convert tensor to PIL Image
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
        transforms.ToPILImage()
    ])
    
    img = inv_normalize(img_tensor.cpu())
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot original image
    ax1.imshow(np.array(img))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot similarity matrix
    im = ax2.imshow(similarity_matrix, cmap='viridis')
    ax2.set_title('49×49 Patch Similarity Matrix')
    ax2.set_xlabel('Patch Index')
    ax2.set_ylabel('Patch Index')
    plt.colorbar(im, ax=ax2, label='Similarity')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def visualize_patches_grid(img_tensor, save_path=None):
    """Visualize the 7x7 grid of patches on the original image"""
    # Convert tensor to PIL Image
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
        transforms.ToPILImage()
    ])
    
    img = inv_normalize(img_tensor.cpu())
    img_array = np.array(img)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display image
    ax.imshow(img_array)
    
    # Add grid lines to represent patches
    # The Conv5 layer of ResNet50 has a 7x7 spatial resolution
    height, width = img_array.shape[:2]
    
    # Draw vertical grid lines
    for i in range(1, 7):
        x = i * width / 7
        ax.axvline(x, color='white', linestyle='-', linewidth=1, alpha=0.7)
        ax.axvline(x, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Draw horizontal grid lines
    for i in range(1, 7):
        y = i * height / 7
        ax.axhline(y, color='white', linestyle='-', linewidth=1, alpha=0.7)
        ax.axhline(y, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add patch labels (numbers) at the center of each patch
    patch_idx = 0
    for i in range(7):
        for j in range(7):
            y = (i + 0.5) * height / 7
            x = (j + 0.5) * width / 7
            ax.text(x, y, str(patch_idx), color='yellow', fontsize=9, 
                    ha='center', va='center', 
                    bbox=dict(boxstyle='round', fc='black', ec='none', alpha=0.5, pad=0.1))
            patch_idx += 1
    
    ax.set_title('Image with 7×7 Patch Grid (49 patches)')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def process_dataset(data_loader, feature_extractor, output_dir, num_samples=100, visualize=True):
    """Process dataset to extract features and compute patch similarity"""
    os.makedirs(output_dir, exist_ok=True)
    features_dir = os.path.join(output_dir, 'features')
    patches_dir = os.path.join(output_dir, 'patches')
    similarity_dir = os.path.join(output_dir, 'similarity')
    visualization_dir = os.path.join(output_dir, 'visualizations')
    tables_dir = os.path.join(output_dir, 'tables')
    html_tables_dir = os.path.join(output_dir, 'html_tables')
    images_dir = os.path.join(output_dir, 'original_images')
    comparison_dir = os.path.join(output_dir, 'comparisons')
    patches_grid_dir = os.path.join(output_dir, 'patches_grid')
    
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)
    os.makedirs(similarity_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(patches_grid_dir, exist_ok=True)
    if visualize:
        os.makedirs(visualization_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(html_tables_dir, exist_ok=True)
    
    # Limit the number of samples to process
    processed_samples = 0
    
    for batch_idx, (images, _) in enumerate(tqdm(data_loader, desc="Processing batches")):
        if processed_samples >= num_samples:
            break
        
        batch_size = images.shape[0]
        remaining = num_samples - processed_samples
        if remaining < batch_size:
            images = images[:remaining]
            batch_size = remaining
        
        # Process batch
        results = feature_extractor.process_batch(images)
        
        # Save results
        for i in range(batch_size):
            idx = processed_samples + i
            
            # Save original image
            img_save_path = os.path.join(images_dir, f'image_{idx}.png')
            save_image(images[i], img_save_path)
            
            # Save features
            feature = results['features'][i].cpu().numpy()
            np.save(os.path.join(features_dir, f'feature_{idx}.npy'), feature)
            
            # Save patches
            patch = results['patches'][i].cpu().numpy()
            np.save(os.path.join(patches_dir, f'patch_{idx}.npy'), patch)
            
            # Save similarity matrix
            similarity = results['similarity'][i]
            np.save(os.path.join(similarity_dir, f'similarity_{idx}.npy'), similarity)
            
            # Create comparison visualization
            comparison_save_path = os.path.join(comparison_dir, f'comparison_{idx}.png')
            create_comparison_visualization(images[i], similarity, comparison_save_path)
            
            # Visualize patches grid
            patches_grid_path = os.path.join(patches_grid_dir, f'patches_grid_{idx}.png')
            visualize_patches_grid(images[i], patches_grid_path)
            
            # Visualize similarity matrix
            if visualize and (idx < 10 or idx % 10 == 0):  # Visualize only a few samples
                vis_save_path = os.path.join(visualization_dir, f'similarity_{idx}.png')
                visualize_similarity(similarity, vis_save_path)
                
                # Visualize with values and save as table
                visualize_similarity_with_values(similarity, vis_save_path)
                table_save_path = os.path.join(tables_dir, f'similarity_table_{idx}.csv')
                save_similarity_table(similarity, table_save_path)
                
                # Save as HTML table
                html_save_path = os.path.join(html_tables_dir, f'similarity_table_{idx}.html')
                save_similarity_html_table(similarity, html_save_path)
        
        processed_samples += batch_size
    
    print(f"Processed {processed_samples} samples.")

def main():
    parser = argparse.ArgumentParser(description='Extract patches from images using ResNet50')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to store the dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to store the output')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to process')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--visualize', action='store_true', help='Visualize similarity matrices')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    
    # Check GPU
    has_gpu = check_gpu()
    use_cuda = has_gpu and not args.no_cuda
    
    # Download dataset
    dataset_path = download_tiny_imagenet(args.data_dir)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        dataset_path, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create feature extractor
    feature_extractor = ResNet50FeatureExtractor(use_cuda=use_cuda)
    
    # Process train dataset
    print("Processing train dataset...")
    train_output_dir = os.path.join(args.output_dir, 'train')
    process_dataset(
        train_loader, 
        feature_extractor, 
        train_output_dir, 
        num_samples=args.num_samples,
        visualize=args.visualize
    )
    
    # Process validation dataset
    print("Processing validation dataset...")
    val_output_dir = os.path.join(args.output_dir, 'val')
    process_dataset(
        val_loader, 
        feature_extractor, 
        val_output_dir, 
        num_samples=args.num_samples,
        visualize=args.visualize
    )
    
    print("Done!")

if __name__ == "__main__":
    main() 