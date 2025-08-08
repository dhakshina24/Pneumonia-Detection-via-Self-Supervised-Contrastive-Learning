import torch
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

    # Load the pretrained DenseNet-121 model
    model = models.densenet121(weights=None)
    pretrained_file = "mimic-chexpert_lr_0.01_bs_128_fd_128_qs_65536.pt"
    pretrained_dict = torch.load(pretrained_file, map_location=device)["state_dict"]

    # Adjust the state_dict keys to match the model's architecture
    arch = "densenet121"
    state_dict={}
    for k, v in pretrained_dict.items():
        if k.startswith("model.encoder_q."):
            k = k.replace("model.encoder_q.", "")
            state_dict[k] = v

    # Check if the classifier weights are present in the pretrained dictionary
    if "model.encoder_q.classifier.weight" in pretrained_dict.keys():
        feature_dim = pretrained_dict["model.encoder_q.classifier.weight"].shape[0]
        in_features = pretrained_dict["model.encoder_q.classifier.weight"].shape[1]

        # Load the model with the adjusted state_dict
        model = models.__dict__[arch](num_classes=feature_dim)
        # Load the state_dict into the model
        model.load_state_dict(state_dict)
        # Remove the original classifier and add a new one
        del model.classifier
        # Add a new classifier with the correct input features
        model.add_module("classifier", torch.nn.Linear(in_features, 2))

        
        for param in model.parameters():
            param.requires_grad = True

        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Fine tune the model
        trainer = Trainer(model, device, num_epochs=10, lr=0.001)
        trainer.train(train_loader)
        print("Model training completed")

        # Evaluate the model
        true_labels, predicted_labels, prediction_probs = trainer.evaluate(test_loader)
        print("Model evaluation completed")