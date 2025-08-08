import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from dataloader import CustomDataLoader
from train_eval_script import Trainer

if __name__ == "__main__":
    
    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    data_loader = CustomDataLoader(dataset_path="./test_data", batch_size=64, train_split=0.8)
    train_loader, test_loader, classes = data_loader.get_data_loaders()
    print("Data loaders created")

    # Load the pretrained DenseNet-121 model
    model = models.resnet50(pretrained=False)
    num_classes = len(classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    # Train the model
    trainer = Trainer(model, device, num_epochs=10, lr=0.001, momentum=0.9)
    trainer.train(train_loader)
    print("Model training completed")

    # Evaluate the model
    true_labels, predicted_labels, prediction_probs = trainer.evaluate(test_loader)
    print("Model evaluation completed")