import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.model.residual_block import ResidualBlock


class Gexpnet(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Gexpnet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.res_block = ResidualBlock(in_channels=8, heads=3)
        res_out = self.res_block.out_channels

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(in_features * res_out, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.conv(x)  
        x = self.res_block(x)  
        x = self.flatten(x)   
        return self.classifier(x)  

    def train_model(self, train_loader, device, epochs=100):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adamax(self.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        for epoch in range(epochs):
            self.train()
            epoch_loss, correct, total = 0.0, 0, 0

            for x_train, y_train in train_loader:
                x_train, y_train = x_train.to(device), y_train.to(device)

                y_pred = self(x_train)
                loss = criterion(y_pred, y_train)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pred = torch.argmax(y_pred, 1)
                true = torch.argmax(y_train, 1) if y_train.ndim == 2 else y_train
                correct += (pred == true).sum().item()
                total += len(y_train)

            avg_loss = epoch_loss / len(train_loader)
            acc = correct / total * 100

            scheduler.step(avg_loss)
            #print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {acc:.2f}% - LR: {self.opt.param_groups[0]['lr']:.6f}")


    def test_model(self, test_loader, device):
        self.eval()
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            correct_all = 0
            for x_test, y_test in test_loader:
                x_test = x_test.to(device)
                y_test = y_test.to(device)

                y_pred = self(x_test)
                _, predicted = torch.max(y_pred.data, 1)
                true = torch.argmax(y_test, 1) if y_test.ndim == 2 else y_test
                correct_all = (predicted == true).float().mean()

            tst_loss = criterion(y_pred, y_test)
            tst_acc = float(correct_all) * 100

        return y_test.cpu(), predicted.cpu(), tst_acc, tst_loss.item()
