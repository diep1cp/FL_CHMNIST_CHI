# FL_CHMNIST_CHI
I have two solutions in each file of code:

1. CHMNIST - MOBILENETV2 + FL :
For this project, I used Federated Learning with the CH-MNIST dataset and trained a MobileNetV2 model using the Flower framework. I split the dataset across three clients to simulate a realistic federated setting. Before running FL, I tested the model locally on one client and was able to get up to 97.6% validation accuracy, which gave me a good baseline. I used focal loss to deal with class imbalance, and I trained the model using SGD with momentum to help avoid overfitting. During federated training, I kept the base layers of MobileNetV2 frozen and ran 10 rounds of FedAvg to see how well the model could learn across all clients without sharing raw data.


2. CHMNIST - CNN + FL :
In this version of the project, I used a custom CNN model instead of MobileNetV2 and applied Federated Learning with the CH-MNIST dataset using the Flower framework. I split the data across multiple clients and made sure each one had samples from all classes. I used focal loss again to handle class imbalance and trained the CNN using an SGD-based optimizer. The model was trained over multiple rounds using FedAvg, and I included global evaluation to track performance across rounds. This setup gave me more control over the architecture compared to using a pretrained model, and helped me see how a lightweight CNN performs in a federated setup.

