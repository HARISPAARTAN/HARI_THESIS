import os
import numpy as np
from PIL import Image, ImageDraw

def create_test_image(size=224, save_path='test_image.jpg'):
    """Create a test image with patterns for diversity analysis"""
    # Create a white background
    img = Image.new('RGB', (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw concentric rectangles with different colors
    for i in range(0, size, 20):
        color = (i % 255, (i * 2) % 255, (i * 3) % 255)
        draw.rectangle([i, i, size-i, size-i], outline=color)
    
    # Draw some diagonal lines
    for i in range(0, size, 40):
        color = ((i * 3) % 255, (i * 2) % 255, i % 255)
        draw.line([(0, i), (i, 0)], fill=color, width=2)
        draw.line([(size-i, size), (size, size-i)], fill=color, width=2)
    
    # Add some circles
    for i in range(40, size//2, 40):
        draw.ellipse([size//2-i, size//2-i, size//2+i, size//2+i], 
                    outline=(0, i % 255, (255-i) % 255))
    
    # Save image
    img.save(save_path)
    print(f"Test image saved to: {os.path.abspath(save_path)}")
    return save_path

if __name__ == "__main__":
    # Create and save a test image
    image_path = create_test_image()
    
    print("\nTo test the trained model on this image, run:")
    print(f"python test_trained_model.py --image {image_path} --model ./trained_model/model.pth") 