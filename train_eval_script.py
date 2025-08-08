import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

class Trainer: 
  def __init__(self, model, device, num_epochs=10, lr=0.001, momentum=0.9):
    self.device = device
    self.model = model.to(self.device)
    self.num_epochs = num_epochs
    self.lr =lr
    self.criterion = nn.CrossEntropyLoss()
    self.momentum = momentum
    # Using SGD optimizer with momentum
    self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
    self.true_labels = []
    self.predicted_labels = []
    self.prediction_probs = []

  
  def train(self, train_loader):
    '''Train the model'''
    print(f"Training on {self.device}")
    self.model.train()

    for epoch in range(self.num_epochs):
      running_loss = 0.0
      train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

      for batch_id, (images, labels) in enumerate(train_bar):
        images = images.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix(loss=running_loss / (batch_id + 1))
      
      epoch_loss = running_loss / len(train_loader)
      print(f"Epoch {epoch+1} Loss: {epoch_loss}")
  

  def evaluate(self, test_loader):
    '''Evaluate the model'''
    self.model.eval()
    correct = 0
    total = 0
    self.true_labels = []
    self.predicted_labels = []
    self.prediction_probs = []

    print("Start Model Evaluation")
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            self.true_labels.extend(labels.tolist())
            self.predicted_labels.extend(predicted.tolist())
            self.prediction_probs.extend(torch.softmax(outputs, dim=1).tolist())

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    self.eval_metrics()

    return self.true_labels, self.predicted_labels, self.prediction_probs
  
  def eval_metrics(self):
    '''Calculate evaluation metrics'''
    report = classification_report(self.true_labels, self.predicted_labels)
    print(report)
    
    # Compute the probabilities for the positive class
    pos_probs = [probs[1] for probs in self.prediction_probs]
    
    # Compute the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(self.true_labels, pos_probs)
    
    # Compute the ROC AUC score
    roc_auc = roc_auc_score(self.true_labels, pos_probs)

    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    # Print the ROC AUC score
    print(f"ROC AUC Score: {roc_auc}")
