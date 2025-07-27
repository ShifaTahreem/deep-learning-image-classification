# deep-learning-image-classification
IMPLEMENT A DEEP LEARNING MODEL FOR IMAGE CLASSIFICATION OR NATURAL LANGUAGE PROCESSING USING TENSORFLOW OR PYTORCH

COMPANY:CODECH IT SOLUTIONS

INTERN ID:CT04DZ1769

DOMAIN:DATA SCIENCE

DURATION:4 WEEKS

MENTOR:NEELA SANTOSH

PROJECT DESCRIPTION:

As part of my internship at CodTech, I was given the opportunity to work on a deep learning project focused on image classification. The objective of the task was to implement a deep learning model using PyTorch that could learn to classify images based on patterns in data. I chose to work with the MNIST dataset, which consists of handwritten digits ranging from 0 to 9. This dataset is a standard benchmark used in computer vision and machine learning for evaluating classification models.

The main goal of this project was to build a complete deep learning workflow, from data loading to model training, testing, and visualization of results. This helped me gain practical experience with key concepts in deep learning such as data preprocessing, neural network architecture, activation functions, loss functions, optimization, and evaluation metrics.

Dataset Used:
I used the MNIST dataset, which contains:

60,000 training images

10,000 testing images
Each image is in grayscale format with a resolution of 28x28 pixels. These images are labeled with the corresponding digit they represent (from 0 to 9).

I used the torchvision.datasets module to load the MNIST dataset directly into the PyTorch framework. The data was transformed using ToTensor() and normalized to help the model learn faster and more accurately.

Model Architecture:

I implemented a simple feedforward neural network using PyTorch. The structure of the model is as follows:

Input Layer: 784 neurons (28x28 pixels flattened into one long vector)

Hidden Layer 1: 128 neurons with ReLU activation

Hidden Layer 2: 64 neurons with ReLU activation

Output Layer: 10 neurons with softmax activation (one for each digit class)

The model was defined by subclassing nn.Module, and a custom forward() function was created to define the flow of data through the network. The model learns by adjusting the weights using backpropagation.

Training the Model:
I used the Adam optimizer, which is known for faster convergence and better results compared to traditional SGD. The loss function used was CrossEntropyLoss, which is suitable for multi-class classification problems.

Epochs: 5

Batch Size: 64

Each epoch involved passing the training data through the network, calculating the loss, and updating the weights. I also tracked the loss at each epoch to monitor the model’s learning progress.

Model Evaluation:
After training, I tested the model on the unseen 10,000 test images. The model achieved a test accuracy of 96.68%, which indicates strong performance in recognizing handwritten digits.

To further analyze the results, I created a confusion matrix using sklearn.metrics and visualized it with the seaborn and matplotlib libraries. The confusion matrix showed how many digits were correctly or incorrectly classified. It helped identify which digits the model found confusing (for example, the digit '5' being confused with '6').

Visualization and Reporting:
I exported the confusion matrix graph as output_graph.png to visually represent the model’s performance. Additionally, I prepared a report file model_accuracy.txt which includes key statistics such as:

Model Name

Framework used (PyTorch)

Dataset used (MNIST)

Final test accuracy

Training settings like batch size and number of epochs

Deployment and Submission:

All the project files — including:

The Jupyter Notebook (.ipynb)

The output_graph.png

The model_accuracy.txt

were uploaded to GitHub as my final submission. This helps in keeping the project accessible and version-controlled, and also serves as a part of my technical portfolio.

What I Learned:
This project provided me with hands-on experience in:

Building neural networks using PyTorch

Preprocessing image data

Training and evaluating a deep learning model

Visualizing results using real-world metrics

Managing files and submissions via GitHub

Working on this internship task gave me a better understanding of how artificial intelligence can be applied to solve practical problems like image recognition. It also helped me get more confident with Python programming, PyTorch syntax, and machine learning workflows.

OUTPUT:

