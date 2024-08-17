import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import getDataloader
from model import getModel

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*inputs.size(0)
        
        epoch_loss = running_loss/len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs-1}, LOss: {epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_accuracy = 100* correct/total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')


def main() :
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    model = getModel()
    train_loader, val_loader = getDataloader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

if __name__ == '__main__' :
    main()
