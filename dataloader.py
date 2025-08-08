import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class CustomDataLoader():
    def __init__(self, dataset_path = "", batch_size=64, train_split = 0.8):
        self.data_path = dataset_path
        self.batch_size = batch_size 
        self.train_split = train_split  

    def get_transforms(self):
        # Define the transformations
        print("Transforming images")

        return transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

    def get_data_loaders(self):
        # Load the dataset
        dataset = ImageFolder(self.data_path, transform=self.get_transforms())
        
        # Calculate the number of samples for train-test split
        total_samples = len(dataset)
        train_samples = int(self.train_split * total_samples)
        test_samples = total_samples - train_samples

        # Split the dataset into train and test sets
        train_set, test_set = torch.utils.data.random_split(dataset, [train_samples, test_samples])

        # Create data loaders for train and test sets
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader, dataset.classes
