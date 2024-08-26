import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MultiLabelBinarizer
import random
from utils import plot_image
from PIL import Image
import numpy as np
import pandas as pd
import os

DATA_CSV = '/scratch/haris.1/CXR8/Data_Entry.csv'
IMAGES_FOLDER = '/scratch/haris.1/CXR8/images/images'

# Define a custom dataset
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, images_folder, all_labels, transform=None):
        self.dataframe = dataframe
        self.images_folder = images_folder
        self.transform = transform

         # Assuming labels are already in the correct format in dataframe
        self.labels = dataframe[all_labels].values
        self.image_paths = dataframe['Finding Labels'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_folder, self.dataframe.iloc[idx, 0])
        try:
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except (OSError, FileNotFoundError) as e:
            print(f"Error opening image {img_name}: {e}")
            image = Image.new("RGB", (224, 224))  # Placeholder image if file is missing
            # Ensure image is a tensor
            image = transforms.ToTensor()(image)
            image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)  # Normalize    


        labels = self.labels[idx]
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize    
])

def preprocess_labels(df):
    # Convert 'Finding Labels' to a list of labels
    df['Finding Labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))
    
    # Create a MultiLabelBinarizer instance
    mlb = MultiLabelBinarizer()
    
    # Fit and transform the labels
    labels = mlb.fit_transform(df['Finding Labels'])
    
    # Create a DataFrame with the new binary labels
    label_df = pd.DataFrame(labels, columns=mlb.classes_, index=df.index)
    
    # Concatenate the original dataframe with the new label columns
    df = pd.concat([df, label_df], axis=1)
    
    return df, mlb.classes_

def createDatasets():
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset = ['Finding Labels'])

    df, all_labels = preprocess_labels(df)

    # cc = df['Finding Labels'].value_counts()
    # print(cc)

    print(df[all_labels].sum())
    #print(df['Finding Labels'].unique())

    X = df.drop(['Finding Labels'] + list(all_labels), axis=1)
    y = df[all_labels]

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    # Create dataset
    train_dataset = ChestXrayDataset(train_df, IMAGES_FOLDER, all_labels, transform=transform)
    val_dataset = ChestXrayDataset(val_df, IMAGES_FOLDER, all_labels, transform=transform)

    return train_dataset, val_dataset

def getDataloader():
    train_dataset, val_dataset = createDatasets()

    #Create data loaders    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    for inputs, labels in train_loader:
        print(f'Inputs shape: {inputs.shape}')
        print(f'Labels shape: {labels.shape}')
        break

    # Display image and label.
    # train_features, train_labels = next(iter(train_loader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    return train_loader, val_loader

def getTestDataloader():
    test_dataset =  datasets.ImageFolder(root='../Data/TB_Chest_Radiography_Database/Test', transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=4)

    return test_loader

if __name__ == '__main__' :
    tds, vds = getDataloader()