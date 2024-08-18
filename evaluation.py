import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from dataloader import getDataloader
from model import getModel

def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set model to evaluate mode
    all_labels = []
    all_preds = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(test_loader.dataset)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Print evaluation results
    print(f'Evaluation Loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Print classification report and confusion matrix
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds))
    print('Confusion Matrix:')
    print(confusion_matrix(all_labels, all_preds))


def main():
    model_path = 'models/final_model.pth'  # Replace with your model's path

    model = getModel()
    model.load_state_dict(torch.load(model_path))

    _, _, test_loader = getDataloader()

    criterion = torch.nn.CrossEntropyLoss()

    evaluate_model(model, test_loader, criterion)


if __name__ == '__main__':
    main()
