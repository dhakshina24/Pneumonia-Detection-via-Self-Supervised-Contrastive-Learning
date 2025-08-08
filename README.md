# Pneumonia Detection via Self-Supervised Contrastive Learning

## Project Overview
This project explores the use of self-supervised contrastive learning for medical image classification. A pretrained self-supervised model is fine-tuned on a labeled chest X-ray dataset to perform pneumonia detection, and its performance is benchmarked against traditional supervised learning models trained from scratch.

The goal is to evaluate whether self-supervised representations learned without human labels can be effectively transferred to downstream medical tasks, potentially reducing reliance on large labeled datasets in healthcare.

## Architecture
- **Self-Supervised Approach**: Pre-trained MoCo v2 model with DenseNet backbone, fine-tuned for pneumonia classification
- **Supervised Baselines**: ResNet and DenseNet architectures trained from scratch
- **Evaluation Framework**: Comprehensive performance comparison and analysis


## Installation
1. **Clone the repository**
```
git clone <repo-url>
```
<br>

2. **Install Dependencies**
```
pip install -r requirements.txt
```
<br>

3. **Dataset Setup**
- Download Data from Kaggle : [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Extract and place in 'data/' directory

<br>

4. **Pre-trained Model Setup**
- Download the pre-trained MoCo model from: [Facebook Research CovidPrognosis](https://github.com/facebookresearch/CovidPrognosis)
- Follow their instructions to obtain the pre-trained weights
- Place the model weights in `models/` directory

<br>

5. **Run the Project**
- **Fine-tuning the self-supervised model:**
```
python ssl_densenet121.py
```

<br>

- **Training and evaluating supervised baselines:**
```
# Train ResNet model 
python resnet50.py

# Train DenseNet model
python densenet121.py
```


## Result
The self-supervised model outperformed traditional supervised learning models for pneumonia detection. The MoCo v2 fine-tuned model achieved the best overall performance, followed by supervised DenseNet and ResNet models.

Detailed classification reports and metrics are available in the [`results/`](results/) folder.

This demonstrates the potential of self-supervised contrastive learning for medical image classification, especially when labeled data is limited.