import torch
import torch.nn as nn
import torch.nn as nn
import torchvision.models as models
from dataloader import CustomDataLoader
from train_eval_script import Trainer

if __name__ == "__main__":
    
    # Set the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    data_loader = CustomDataLoader(dataset_path="./test_data", batch_size=64, train_split=0.8)
    train_loader, test_loader, classes = data_loader.get_data_loaders()
    print("Data loaders created")

    # Load the DenseNet model without weights
    model = models.densenet121(pretrained=False)
    num_classes = len(classes)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    # Train the model
    trainer = Trainer(model, device, num_epochs=10, lr=0.001, momentum=0.9)
    trainer.train(train_loader)
    print("Model training completed")

    # Evaluate the model
    true_labels, predicted_labels, prediction_probs = trainer.evaluate(test_loader)
    print("Model evaluation completed")