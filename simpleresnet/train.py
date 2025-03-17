import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import time
from tqdm import tqdm
from dataset import get_dataloaders
from model import get_resnet50

def train_model(epochs=5, batch_size=128, lr=0.001):
    # Initialize WandB
    wandb.init(project="tiny-imagenet-resnet50", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "architecture": "ResNet-50",
        "dataset": "Tiny ImageNet"
    })
    
    # Detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU not found, running on CPU!")

    # Move model to GPU
    model = get_resnet50().to(device)
    
    # Load dataset
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    
    # Move loss function to GPU
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("🚀 Training starts...")
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0
        start_time = time.perf_counter()  # Track time for IPS

        # Progress bar for visualization
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU

            optimizer.zero_grad()
            outputs = model(images) # h x w x d
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Calculate IPS (Iterations per Second)
            elapsed_time = time.perf_counter() - start_time
            ips = (batch_idx + 1) / elapsed_time if elapsed_time > 0 else 0

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total, IPS=f"{ips:.2f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        print(f"🟢 Epoch {epoch+1} Completed | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}% | IPS: {ips:.2f}")

        # Log metrics to WandB
        wandb.log({"Loss": epoch_loss, "Accuracy": epoch_acc, "IPS": ips})

    print("🎉 Training complete.")
    wandb.finish()


## pesudocode for loss
def new_loss(outputs):
    x = einops.reshape ("h w c -> (hw) c", outputs)
    x_T = einops.reshape( "hw c -> c hw")
    matric = einops( 'hw1 c, c hw2 -> hw1 hw2', x, x_T)
    torch.sort(martic, dim=-1)
    ones_matrics

    return ((ones_matrics - matric)**2).mean()



if __name__ == "__main__":
    train_model(epochs=5)
