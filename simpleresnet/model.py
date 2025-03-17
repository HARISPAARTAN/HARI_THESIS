import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights  # Import weights

def get_resnet50(num_classes=200):
    # Load ResNet50 with pre-trained weights
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # Modify final fully connected layer for 200 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
