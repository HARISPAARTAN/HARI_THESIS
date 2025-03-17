import os
from download_data import download_tiny_imagenet
from train import train_model

if __name__ == "__main__":
    # Ensure dataset is available
    if not os.path.exists("tiny-imagenet-200"):
        download_tiny_imagenet()

    # Start training
    train_model(epochs=5)
