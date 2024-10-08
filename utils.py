import torch
import os
import matplotlib.pyplot as plt

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f'Model loaded from {path}')
    return model

def plot_metrics(train_loss, val_loss, val_accuracy):
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracy, 'g', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Directory {directory} created.')
    else:
        print(f'Directory {directory} already exists.')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_plot_as_image(fig, path):
    fig.savefig(path)
    print(f'Plot saved as {path}')

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std

def plot_image(image, title=None):
    image = image.numpy()
    image = image.transpose(1,2,0)
    plt.imshow(image, cmap='gray')
    if title is not None :
        plt.title(title)
    plt.show()