# Simple Contrastive Representation Learning Technique to Improve Task-Agnostic Model Performance with Unlabeled Dataset
Self-supervised (Semi-supervised) learning technique which trains a CNN model with unlabeled data to improve a task-agnostic/task-specific representation learning.
Quick look at original idea and concept: 
  - SimpleCRLv1 paper here <a href="https://arxiv.org/pdf/2002.05709.pdf">https://arxiv.org/pdf/2002.05709.pdf</a> 
  - SimpleCRLv2 paper here <a href='https://arxiv.org/pdf/2006.10029.pdf'>https://arxiv.org/pdf/2006.10029.pdf</a>
  - Original authors implementation <a href = "https://github.com/google-research/simclr">https://github.com/google-research/simclr</a>
  
  
| Contrastive Representation Learning            Pretrained weights     |                                                |
|----------------------------------------------|------------------------|
| Logistic regression with logit dimension 10  | [LogisticRegression_10](https://drive.google.com/file/d/1eN9RHzBn69Il9dxy0zIARKrnJhxo_SxE/view?usp=sharing) |
| Logistic regression with logit dimension 20  | [LogisticRegression_20](https://drive.google.com/file/d/1ZNvXnofGR4StzBFoewr65BqzIokrrDyO/view?usp=sharing) |
| Logistic regression with logit dimension 50  | [LogisticRegression_50](https://drive.google.com/file/d/1Kz2HGVnlTl_h0qXJwOijzvANTYCrapCO/view?usp=sharing) |
| Logistic regression with logit dimension 100 | [LogisticRegression_100](https://drive.google.com/file/d/10jyzLyr-q02RfqyFAOLdadwAKhLWP5pi/view?usp=sharing) |
| Logistic regression with logit dimension 200 | [LogisticRegression_200](https://drive.google.com/file/d/1I91fb-JB17nZpgnM7vZbdR8uzN-vYxUb/view?usp=sharing) |
| Logistic regression with logit dimension 500 | [LogisticRegression_500](https://drive.google.com/file/d/1HIPVP2NwTKc40ez4VjrQ-D4Dpvtl0qJH/view?usp=sharing) |

# Performance of simple ResNet50 model on classification task
Current repository contains all necessary function and pretrained models for only ResNet50 model. User can freely modefy the code and try larger or smaller models.
We also used STL10 dataset to perform self-supervised training over unlabeled image data. STL10 dataset is already provided in `data/` directory. Following table contains test accuracy for various number of images per class.
  
| <b> Images per class <b>   |<b> Test Acc.<b>  | 
|----------------------------|------------------|
| 10                         |       62.7%      |
| 20                         |       68.6%      |
| 50                         |       74.4%      |
| 100                        |       77.2%      |
| 200                        |       79.1%      |
| 500                        |       81.3%      |
