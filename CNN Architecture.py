"""
# What is a Convolutional Neural Network (CNN), and why is it used for image processing?
A CNN is a type of deep learning model designed to process grid-like data such as images. It automatically learns spatial hierarchies of features, making it ideal for tasks like image recognition.

# What are the key components of a CNN architecture?
1. Convolutional layers
2. Pooling layers
3. Fully connected layers
4. Activation functions
Example: Image -> Conv -> Pool -> FC -> Output

# What is the role of the convolutional layer in CNNs?
It extracts spatial features by applying filters to the input image, preserving spatial relationships.

# What is a filter (kernel) in CNNs?
A small matrix that slides over the input to perform element-wise multiplications and sums, detecting features like edges or patterns.
Example: A 3x3 filter detects vertical edges.

# What is pooling in CNNs, and why is it important?
Pooling reduces the spatial size of feature maps, lowering computation and preventing overfitting.
Example: Max pooling retains the highest value in each region.

# What are the common types of pooling used in CNNs?
1. Max pooling: Takes the max value in a region.
2. Average pooling: Computes the average of a region.
Example: Max pooling: [[1, 2], [3, 4]] -> 4

# How does the backpropagation algorithm work in CNNs?
It computes the gradient of the loss function w.r.t. model parameters (weights/filters) and updates them using gradient descent.

# What is the role of activation functions in CNNs?
They introduce non-linearity, enabling the network to learn complex patterns.
Example: ReLU(x) = max(0, x)

# What is the concept of receptive fields in CNNs?
The receptive field refers to the region of the input image that influences a particular neuron in a layer.

# Explain the concept of tensor space in CNNs.
It represents the multi-dimensional structure of data as it passes through layers, e.g., (batch_size, height, width, channels).

# What is LeNet-5, and how does it contribute to the development of CNNs?
LeNet-5, introduced by Yann LeCun, was one of the first CNNs for digit recognition, demonstrating CNNs' effectiveness.

# What is AlexNet, and why was it a breakthrough in deep learning?
AlexNet, a deeper CNN, won the 2012 ImageNet competition by utilizing GPUs and ReLU activations.

# What is VGGNet, and how does it differ from AlexNet?
VGGNet uses smaller (3x3) filters and deeper architectures for more refined feature extraction compared to AlexNet.

# What is GoogLeNet, and what is its main innovation?
GoogLeNet introduced the Inception module, allowing efficient multi-scale feature extraction within the network.

# What is ResNet, and what problem does it solve?
ResNet introduces residual connections to solve the vanishing gradient problem, enabling very deep networks.

# What is DenseNet, and how does it differ from ResNet?
DenseNet connects all layers, promoting feature reuse and improving efficiency over ResNet's residual connections.

# What are the main steps involved in training a CNN from scratch?
1. Prepare data (preprocessing and augmentation).
2. Define CNN architecture.
3. Initialize weights.
4. Train with forward and backward propagation.
5. Validate and tune hyperparameters.
6. Test and evaluate performance.

"""


# Implement a basic convolution operation using a filter and a 5x5 image (matrix)
import numpy as np
image = np.array([[1, 2, 3, 0, 1],
                  [4, 5, 6, 1, 0],
                  [7, 8, 9, 0, 1],
                  [0, 1, 2, 3, 4],
                  [1, 0, 1, 2, 3]])
filter = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])
output = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        region = image[i:i+3, j:j+3]
        output[i, j] = np.sum(region * filter)
print(output)

# Implement max pooling on a 4x4 feature map with a 2x2 window
feature_map = np.array([[1, 3, 2, 1],
                        [4, 6, 5, 0],
                        [7, 8, 9, 4],
                        [2, 3, 1, 5]])
pooled = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        region = feature_map[i*2:(i+1)*2, j*2:(j+1)*2]
        pooled[i, j] = np.max(region)
print(pooled)

# Implement the ReLU activation function on a feature map
relu_feature_map = np.maximum(0, feature_map)
print(relu_feature_map)

# Create a simple CNN model with one convolutional layer and a fully connected layer, using random data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    Flatten(),
    Dense(10, activation='softmax')
])
model.summary()

# Generate a synthetic dataset using random noise and train a simple CNN model on it
from tensorflow.keras.utils import to_categorical
X_train = np.random.random((100, 32, 32, 3))
y_train = to_categorical(np.random.randint(10, size=100), 10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=10)

# Create a simple CNN using Keras with one convolution layer and a max-pooling layer
from tensorflow.keras.layers import MaxPooling2D
cnn = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2))
])
cnn.summary()

# Write a code to add a fully connected layer after the convolution and max-pooling layers in a CNN
cnn.add(Flatten())
cnn.add(Dense(10, activation='softmax'))
cnn.summary()

# Write a code to add batch normalization to a simple CNN model
from tensorflow.keras.layers import BatchNormalization
cnn = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2))
])

# Write a code to add dropout regularization to a simple CNN model
from tensorflow.keras.layers import Dropout
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))

# Write a code to print the architecture of the VGG16 model in Keras
from tensorflow.keras.applications import VGG16
vgg16 = VGG16(weights=None, input_shape=(224, 224, 3))
vgg16.summary()

# Write a code to plot the accuracy and loss graphs after training a CNN model
history = model.fit(X_train, y_train, epochs=5)
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.legend()
plt.show()

# Write a code to print the architecture of the ResNet50 model in Keras
from tensorflow.keras.applications import ResNet50
resnet50 = ResNet50(weights=None, input_shape=(224, 224, 3))
resnet50.summary()

# Write a code to train a basic CNN model and print the training loss and accuracy after each epoch
for epoch in range(5):
    history = model.fit(X_train, y_train, epochs=1, verbose=0)
    print(f"Epoch {epoch + 1}, Loss: {history.history['loss'][0]}, Accuracy: {history.history['accuracy'][0]}")
