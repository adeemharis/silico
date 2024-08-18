from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define transformations
def createDatasets():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),        # Resize image to 128x128
        transforms.ToTensor(),                # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize (mean, std) for grayscale images
    ])

    # Create dataset
    dataset = datasets.ImageFolder(root='Data/TB_Chest_Radiography_Database', transform=transform)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

def getDataloader():
    train_dataset, val_dataset, test_dataset = createDatasets()

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
