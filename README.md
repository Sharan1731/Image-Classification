# Convolutional Deep Neural Network for Image Classification

## Problem Statement and Dataset

The objective of this project is to create a CNN that can categorize images of fashion items from the Fashion MNIST dataset. This dataset includes grayscale images of clothing and accessories such as T-shirts, trousers, dresses, and footwear. The task is to accurately predict the correct category for each image while ensuring the model is efficient and robust.

1.Training data: 60,000 images

2.Test data: 10,000 images

3.Classes: 10 fashion categories

The CNN consists of multiple convolutional layers with activation functions, followed by pooling layers, and ends with fully connected layers to output predictions for all 10 categories.

## Neural Network Model

<img width="1746" height="732" alt="image" src="https://github.com/user-attachments/assets/15b97d77-5d7e-4d84-abd9-56069c55d2e5" />


## DESIGN STEPS
### STEP 1:
Import the necessary libraries such as NumPy, Matplotlib, and PyTorch.

### STEP 2:
Load and preprocess the dataset:
Resize images to a fixed size (128×128). 
Normalize pixel values to a range between 0 and 1. 
Convert labels into numerical format if necessary.

### STEP 3:
Define the CNN Architecture, which includes:

Input Layer: Shape (8,128,128) 
Convolutional Layer 1: 8 filters, kernel size (16×16), ReLU activation Max-Pooling Layer 1: Pool size (2×2) Convolutional Layer 2: 24 filters, kernel size (8×8), ReLU activation Max-Pooling Layer 2: Pool size (2×2) Fully Connected (Dense) Layer: First Dense Layer with 256 neurons Second Dense Layer with 128 neurons Output Layer for classification

### STEP 4:
Define the loss function (e.g., Cross-Entropy Loss for classification) and optimizer (e.g., Adam or SGD).

### STEP 5:
Train the model by passing training data through the network, calculating the loss, and updating the weights using backpropagation.

### STEP 6:
Evaluate the trained model on the test dataset using accuracy, confusion matrix, and other performance metrics.

### STEP 7:
Make predictions on new images and analyze the results.



## PROGRAM

## PROGRAM

### Name:SHARAN G
### Register Number:212223230203
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


```

```python
model = CNNClassifier()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


```

```python
def train_model(model, train_loader, num_epochs=3):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: SHARAN G')
        print('Register Number: 212223230203')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="429" height="238" alt="Screenshot 2026-03-17 094511" src="https://github.com/user-attachments/assets/71d43d19-b718-487c-9eee-5f8ca8cb848d" />



### Confusion Matrix
<img width="821" height="627" alt="Screenshot 2026-03-17 094645" src="https://github.com/user-attachments/assets/3d519fed-2c86-4b8a-ba4d-ffc56231ef2b" />




### Classification Report
<img width="499" height="359" alt="Screenshot 2026-03-17 094654" src="https://github.com/user-attachments/assets/6d03f2ec-3b5b-4696-a1bb-c8ef4c9b7ee6" />




### New Sample Data Prediction

<img width="456" height="467" alt="Screenshot 2026-03-17 094702" src="https://github.com/user-attachments/assets/ba432ee7-aefd-4103-b785-83334cb93bfe" />


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
