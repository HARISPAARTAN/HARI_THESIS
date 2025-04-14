import os
import zipfile
import requests
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        if split == 'train':
            # Process train data
            self._process_train_data()
        elif split == 'val':
            # Process validation data
            self._process_val_data()
        else:
            raise ValueError("Split must be 'train' or 'val'")
            
    def _process_train_data(self):
        train_dir = os.path.join(self.root_dir, 'train')
        for class_dir in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_dir)
            if os.path.isdir(class_path):
                images_dir = os.path.join(class_path, 'images')
                if os.path.exists(images_dir):
                    for img_file in os.listdir(images_dir):
                        if img_file.endswith('.JPEG'):
                            self.image_paths.append(os.path.join(images_dir, img_file))
                            self.labels.append(class_dir)
    
    def _process_val_data(self):
        val_dir = os.path.join(self.root_dir, 'val')
        val_images_dir = os.path.join(val_dir, 'images')
        val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        
        # Create a mapping from image filename to class
        img_to_class = {}
        with open(val_annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_file, class_id = parts[0], parts[1]
                    img_to_class[img_file] = class_id
        
        # Add validation images
        for img_file in os.listdir(val_images_dir):
            if img_file in img_to_class:
                self.image_paths.append(os.path.join(val_images_dir, img_file))
                self.labels.append(img_to_class[img_file])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # If image is corrupted, return a blank image
            image = Image.new('RGB', (64, 64))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label


def download_tiny_imagenet(root_dir):
    """
    Download and extract Tiny ImageNet dataset
    """
    # Create directory if it doesn't exist
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # URL for Tiny ImageNet
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_file_path = os.path.join(root_dir, "tiny-imagenet-200.zip")
    extract_dir = root_dir
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(os.path.join(root_dir, "tiny-imagenet-200")):
        print(f"Downloading Tiny ImageNet dataset to {zip_file_path}...")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_file_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print("Extracting Tiny ImageNet dataset...")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print("Tiny ImageNet dataset downloaded and extracted successfully.")
    else:
        print("Tiny ImageNet dataset already exists.")
    
    return os.path.join(root_dir, "tiny-imagenet-200")


def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for Tiny ImageNet
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TinyImageNetDataset(data_dir, split='train', transform=transform)
    val_dataset = TinyImageNetDataset(data_dir, split='val', transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader 