# Diverse Patch Learning with ResNet50

This project trains a ResNet50-based model to learn diverse patch representations using a specialized diversity loss function.

## Overview

The model:
- Uses a pretrained ResNet50 as a backbone (frozen)
- Adds a trainable projection layer (2048 → 512 channels)
- Computes a patch similarity matrix for the 7×7 feature maps
- Applies diversity loss to minimize similarity between patches

The diversity loss function encourages patches in the feature map to be different from each other, leading to more discriminative representations.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model on Tiny ImageNet dataset:

```bash
python trainable_model.py --data_dir ./data --epochs 10
```

Options:
- `--data_dir`: Directory to store dataset (default: `./data`)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--output_dir`: Output directory (default: `./trained_model`)
- `--no_cuda`: Disable CUDA

### Testing

Test the trained model on an image:

```bash
python test_trained_model.py --image path/to/image.jpg --model path/to/model.pth
```

Options:
- `--image`: Path to test image (required)
- `--model`: Path to trained model weights (required)
- `--output_dir`: Output directory (default: `./results`)
- `--no_cuda`: Disable CUDA

## Output

- **Similarity Matrix**: 49×49 matrix showing the similarity between each patch
- **Diversity Metric**: Average off-diagonal similarity (lower is better)
- **Visualization**: Heatmap showing the patch similarity matrix

## How It Works

1. **Feature Extraction**: ResNet50 extracts features from input images
2. **Projection**: A trainable layer projects the features to a new space
3. **Patch Extraction**: The 7×7 feature maps are reshaped into 49 patches
4. **Similarity Computation**: Cosine similarity is computed between all patches
5. **Diversity Loss**: The loss function penalizes high similarity between different patches

During training, the model learns to make patches more diverse, resulting in a lower off-diagonal similarity in the patch similarity matrix. 