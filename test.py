import torch
from dataloader import getTestDataloader
from model import getModel
from sklearn.metrics import confusion_matrix, classification_report
from utils import load_model

def test_model(model, test_loader, criterion):
    model.eval()  # Set model to evaluate mode
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Print classification report and confusion matrix
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds))
    print('Confusion Matrix:')
    print(confusion_matrix(all_labels, all_preds))


def main():
    batch_size = 32
    model_path = 'models/final_model.pth'  # Replace with your model's path

    model = getModel()
    model = load_model(model, model_path)

    test_loader = getTestDataloader()

    criterion = torch.nn.CrossEntropyLoss()

    test_model(model, test_loader, criterion)


if __name__ == '__main__':
    main()
