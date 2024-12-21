import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import os
from tqdm import tqdm
from torchsummary import summary

from config import *
from dataset import CIFAR10Dataset
from model import CIFAR10Net
from utils import get_accuracy

def train():
    # Set device
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    
    # Create datasets
    train_dataset = CIFAR10Dataset(
        root="./data",
        train=True,
        transform=train_transforms
    )
    
    test_dataset = CIFAR10Dataset(
        root="./data",
        train=False,
        transform=test_transforms
    )
    
    # Create data loaders with more workers and pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize model, optimizer, and loss function
    model = CIFAR10Net(NUM_CLASSES).to(device)
    
    # Print model summary
    print("\nModel Architecture:")
    summary(model, (3, 32, 32))
    print(f"\nTotal parameters: {model.get_num_params():,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate training and testing accuracy
        train_accuracy, train_loss = get_accuracy(model, train_loader, device)
        test_accuracy, test_loss = get_accuracy(model, test_loader, device)
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print("-" * 50)
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), "best_model.pth")
        else:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    train() 