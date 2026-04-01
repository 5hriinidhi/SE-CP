import time
import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    """
    Standard PyTorch trainer for AutoNAS candidates.
    Supports Adam optimizer, CosineAnnealingLR, and best-model checkpointing.
    """
    def __init__(self, epochs: int = 20, lr: float = 1e-3, device: str = None):
        self.epochs = epochs
        self.lr = lr
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def train(self, model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> dict:
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        best_val_acc = -1.0
        
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_loss = total_loss / max(1, total)
            train_acc = correct / max(1, total)
            
            # Validation
            val_acc, val_loss = self._validate(model, val_loader, criterion)
            
            # Save best candidate to /tmp/ as requested
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), '/tmp/best_candidate.pth')
            
            scheduler.step()
            
        training_duration_s = time.time() - start_time
        
        return {
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "train_accuracy": float(train_acc),
            "train_loss": float(train_loss),
            "epochs_trained": self.epochs,
            "training_duration_s": float(training_duration_s)
        }

    def _validate(self, model: nn.Module, loader: torch.utils.data.DataLoader, criterion: nn.Module) -> tuple:
        """
        Evaluate model on validation set.
        Returns (accuracy, average_loss).
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_acc = correct / max(1, total)
        avg_loss = total_loss / max(1, total)
        return (avg_acc, avg_loss)
