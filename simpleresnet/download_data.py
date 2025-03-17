import os
import requests
import zipfile

def download_tiny_imagenet(data_dir="tiny-imagenet-200"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = "tiny-imagenet-200.zip"
    
    if not os.path.exists(data_dir):
        print("Downloading Tiny ImageNet dataset...")
        r = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)
        print("Download and extraction complete.")
    else:
        print("Dataset already downloaded.")

if __name__ == "__main__":
    download_tiny_imagenet()
