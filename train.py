import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import getDataloader
from model import getModel
from utils import save_model, plot_metrics, ensure_dir

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    train_loss = []
    val_loss = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss.append(epoch_loss)
        print(f'Epoch {epoch}/{num_epochs-1}, Loss: {epoch_loss:.4f}')

        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_accuracy = 100 * correct / total
        val_loss.append(epoch_val_loss)
        val_accuracy.append(epoch_val_accuracy)
        print(f'Validation Loss: {epoch_val_loss:.4f}, Accuracy: {epoch_val_accuracy:.2f}%')

    # Plot the metrics
    plot_metrics(train_loss, val_loss, val_accuracy)

    # Save the final model
    ensure_dir('models')
    save_model(model, 'models/final_model.pth')

def main():
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    model = getModel()
    train_loader, val_loader, _ = getDataloader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

if __name__ == '__main__':
    main()
