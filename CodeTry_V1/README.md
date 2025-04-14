# Tiny ImageNet ResNet50 Feature Extraction

This project extracts patches from images in the Tiny ImageNet dataset using ResNet50's Conv5 layer and computes similarity between patches.

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

- `data_loader.py`: Functions to download and load Tiny ImageNet dataset
- `feature_extractor.py`: ResNet50 feature extractor and patch similarity computation
- `main.py`: Main script to run the extraction process
- `test_visualization.py`: Script to test the visualization features
- `test_custom_image.py`: Script to test with a custom image
- `requirements.txt`: Required packages

## Directory Structure

After running the code, the following directory structure will be created:

```
./data
└── tiny-imagenet-200
    ├── train
    ├── val
    └── ...

./output
├── train
│   ├── features
│   ├── patches
│   ├── similarity
│   ├── visualizations
│   ├── tables
│   ├── html_tables
│   ├── original_images
│   └── comparisons
└── val
    ├── features
    ├── patches
    ├── similarity
    ├── visualizations
    ├── tables
    ├── html_tables
    ├── original_images
    └── comparisons
```

## How to Use

Run the main script with default parameters:

```bash
python main.py
```

### Command Line Arguments

- `--data_dir`: Directory to store the dataset (default: `./data`)
- `--output_dir`: Directory to store the output (default: `./output`)
- `--batch_size`: Batch size (default: 16)
- `--num_samples`: Number of samples to process (default: 100)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--visualize`: Visualize similarity matrices
- `--no_cuda`: Disable CUDA

Example with custom parameters:

```bash
python main.py --data_dir ./my_data --output_dir ./results --batch_size 32 --num_samples 500 --visualize
```

### Quick Test

To quickly test the visualization features without processing the full dataset:

```bash
python test_visualization.py
```

This downloads a sample image from the internet and performs the full pipeline on it.

### Test with Your Own Image

You can also process and visualize a single custom image of your choice:

```bash
python test_custom_image.py --image /path/to/your/image.jpg
```

Additional options:
- `--output_dir`: Directory to save outputs (default: `./custom_image_test`)
- `--no_cuda`: Disable CUDA

The script will:
1. Save the original unprocessed image
2. Extract features and patches using ResNet50
3. Compute patch similarity
4. Generate all visualizations (heatmaps, annotated values, tables, comparison)
5. Save all outputs to the specified directory

## Outputs

- **Original Images**: Original input images (in `original_images` directory)
- **Features**: Extracted ResNet50 Conv5 layer features (shape: [2048, 7, 7])
- **Patches**: Extracted patches from features (shape: [49, 2048])
- **Similarity**: Similarity matrices between patches (shape: [49, 49])
- **Comparisons**: Side-by-side visualizations of original images and their similarity matrices
- **Visualizations**: Visualizations of similarity matrices
  - Standard heatmap visualizations
  - Annotated visualizations with numerical values (7x7 section)
- **Tables**: CSV files with patch similarity values
- **HTML Tables**: Interactive HTML tables with color-coded similarity values

## Viewing Results

### Original Images and Comparisons

- **Original Images**: The original input images are saved in the `original_images` directory
- **Patch Grid**: The original image with the 7×7 patch grid overlay showing each of the 49 patches that correspond to the ResNet50 Conv5 layer output (in `patches_grid` directory)
- **Comparisons**: Side-by-side comparisons of original images and their similarity matrices are saved in the `comparisons` directory

### Visualizations

Check the `visualizations` directory for heatmap visualizations showing the full 49×49 similarity matrix. Each cell in this matrix represents the similarity between one patch and another patch in the image.

Files with `_annotated` suffix show a 7×7 subset of the similarity matrix with numerical values shown directly on the cells.

### Similarity Matrix

The similarity matrix is a 49×49 matrix where:
- Each row and column corresponds to one of the 49 patches from the image
- The value at position (i, j) represents how similar patch i is to patch j
- Higher values (closer to 1) indicate greater similarity between patches
- The diagonal is always 1.0 as each patch is perfectly similar to itself
- The matrix is symmetric since the similarity between patch i and j is the same as between j and i

### Similarity Tables

For a more detailed view of similarity values:

1. **CSV Tables**: Located in the `tables` directory - can be opened with any spreadsheet software
2. **HTML Tables**: Located in the `html_tables` directory - open these files in any web browser to see color-coded similarity values

## GPU Usage

The code automatically detects and uses a GPU if available. To disable GPU usage, use the `--no_cuda` flag.

## Feature Extraction vs. Training

This project includes two different approaches:

1. **Feature Extraction (Non-trainable)**:
   - Uses a pre-trained ResNet50 model with fixed weights
   - Extracts features from the Conv5 layer and computes similarity between patches
   - Files: `feature_extractor.py`, `main.py`
   - No training involved

2. **Trainable Model with Diversity Loss**:
   - Uses ResNet50 backbone with a trainable projection layer
   - Trains the model to maximize diversity between patches
   - Files: `trainable_model.py`, `test_trained_model.py`

### Training for Diverse Patches

The trainable model uses a diversity loss function that encourages patches in the feature map to be different from each other. This is achieved by:

1. Extracting patches from the feature maps
2. Computing cosine similarity between all pairs of patches
3. Minimizing the similarity between different patches (making the off-diagonal elements of the similarity matrix as close to zero as possible)

To train the model:

```bash
python trainable_model.py --data_dir ./data --epochs 10 --batch_size 32 --output_dir ./trained_model
```

Additional options:
- `--lr`: Learning rate (default: 0.001)
- `--no_cuda`: Disable CUDA

### Testing Trained Model

To test the trained model on a custom image and compare it with the pretrained model:

```bash
python test_trained_model.py --image path/to/image.jpg --model ./trained_model/model.pth
```

This will:
1. Load both the pretrained and trained models
2. Extract features and compute similarity matrices for both models
3. Compare the similarity matrices to visualize the difference
4. Calculate a diversity metric for both models

The output will include:
- Diversity scores for both models
- A visualization comparing the similarity matrices

## Original Feature Extraction 