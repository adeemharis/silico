from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from utils import get_mean_std

# Define transformations
transform = transforms.Compose([
        transforms.Resize((512, 512)),        # Resize image to 128x128
        transforms.ToTensor(),                # Convert image to tensor
        #transforms.Normalize((0.5,), (0.5,))  # Normalize (mean, std) for grayscale images
    ])

def createDatasets():
    # Create dataset
    dataset = datasets.ImageFolder(root='../Data/TB_Chest_Radiography_Database/Train', transform=transform)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.3 * len(dataset))

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

def getDataloader():
    train_dataset, val_dataset = createDatasets()

    #Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader

def getTestDataloader():
    test_dataset =  datasets.ImageFolder(root='../Data/TB_Chest_Radiography_Database/Test', transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=4)

    return test_loader