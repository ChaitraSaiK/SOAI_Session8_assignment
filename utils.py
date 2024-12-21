import torch
from typing import Tuple

def get_accuracy(model: torch.nn.Module, 
                data_loader: torch.utils.data.DataLoader, 
                device: str) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)
    return accuracy, avg_loss 