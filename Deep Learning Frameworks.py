"""
# What is TensorFlow 2.0, and how is it different from TensorFlow 1.x?
TensorFlow 2.0 simplifies deep learning with eager execution, `tf.function` for graph building, and integrated Keras API.
Example: No more separate sessions; code is Pythonic.

# How do you install TensorFlow 2.0?
Install via pip:
!pip install tensorflow==2.0.0

# What is the primary function of the tf.function in TensorFlow 2.0?
`tf.function` converts Python functions into TensorFlow graphs for better performance.
Example:
@tf.function
def add(a, b):
    return a + b

# What is the purpose of the Model class in TensorFlow 2.0?
It defines and trains models using layers and provides methods like `fit`, `evaluate`, and `predict`.
Example:
from tensorflow.keras import Model

# How do you create a neural network using TensorFlow 2.0?
# Use Keras layers:
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([Dense(32, activation='relu'), Dense(1, activation='sigmoid')])

# What is the importance of Tensor Space in TensorFlow?
It defines data dimensions (shape, rank, type) for efficient computation and hardware acceleration.
Example: Tensor shape impacts model compatibility.

# How can TensorBoard be integrated with TensorFlow 2.0?
# Use the `tf.summary` API to log metrics during training.
Example:
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
model.fit(X_train, y_train, epochs=10, callbacks=[tensorboard_callback])

# What is the purpose of TensorFlow Playground?
An interactive tool to visualize and understand neural networks' behavior for beginners.

# What is Netron, and how is it # Useful for deep learning models?
Netron is a tool for visualizing neural network models, supporting formats like TensorFlow, ONNX, and PyTorch.

# What is the difference between TensorFlow and PyTorch?
TensorFlow emphasizes static computation graphs (with `tf.function`), while PyTorch # Uses dynamic computation graphs by default.

# How do you install PyTorch?
Install via pip:
!pip install torch torchvision

# What is the basic structure of a PyTorch neural network?
Define a class inheriting from `torch.nn.Module`:
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc(x)

# What is the significance of tensors in PyTorch?
Tensors are multidimensional arrays enabling GPU acceleration for deep learning computations.

# What is the difference between torch.Tensor and torch.cuda.Tensor in PyTorch?
`torch.Tensor` runs on the CPU by default, while `torch.cuda.Tensor` runs on the GPU for faster computation.

# What is the purpose of the torch.optim module in PyTorch?
It provides optimizers like SGD and Adam to update model weights during training.
Example:
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# What are some common activation functions # Used in neural networks?
ReLU, Sigmoid, Tanh, Softmax.
Example: ReLU(x) = max(0, x).

# What is the difference between torch.nn.Module and torch.nn.Sequential in PyTorch?
`nn.Module` provides flexibility for custom forward passes, while `nn.Sequential` chains layers sequentially.
Example:
model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

# How can you monitor training progress in TensorFlow 2.0?
# Use callbacks like `ModelCheckpoint` and `TensorBoard`.
Example:
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# How does the Keras API fit into TensorFlow 2.0?
Keras is integrated as the high-level API for building, training, and evaluating models.

# What is an example of a deep learning project that can be implemented using TensorFlow 2.0?
Image classification using CNNs.
Example:
from tensorflow.keras.applications import ResNet50
model = ResNet50(weights='imagenet')

# What is the main advantage of using pre-trained models in TensorFlow and PyTorch?
Pre-trained models reduce training time and require less data by leveraging learned features.
Example: Transfer learning with ResNet for image classification.

"""

# How do you install and verify that TensorFlow 2.0 was installed successfully?
#Install TensorFlow 2.0 using pip and verify the version:
!pip install tensorflow==2.0.0
import tensorflow as tf
print(tf.__version__)  # Should print '2.0.0'

# How can you define a simple function in TensorFlow 2.0 to perform addition?
## Use `tf.function` for better performance:
@tf.function
def add(a, b):
    return a + b
result = add(3, 5)  # Returns 8

# How can you create a simple neural network in TensorFlow 2.0 with one hidden layer?
# Use the Keras Sequential API:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),  # Hidden layer
    Dense(1, activation='sigmoid')  # Output layer
])
model.compile(optimizer='adam', loss='binary_crossentropy')

# How can you visualize the training progress using TensorFlow and Matplotlib?
#Fit the model and plot training metrics:
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()

# How do you install PyTorch and verify the PyTorch installation?
#Install PyTorch and verify:
!pip install torch torchvision
import torch
print(torch.__version__)  # Verifies PyTorch installation

# How do you create a simple neural network in PyTorch?
#Define a model using `nn.Module`:
import torch.nn as nn
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # Hidden layer
        self.fc2 = nn.Linear(16, 1)  # Output layer
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
model = SimpleNet()

# How do you define a loss function and optimizer in PyTorch?
# Use `torch.nn` for the loss and `torch.optim` for the optimizer:
loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# How do you implement a custom loss function in PyTorch?
#Define a custom loss as a function:
import torch.nn.functional as F
def custom_loss(output, target):
    return torch.mean((output - target)**2)  # Mean Squared Error example

# How do you save and load a TensorFlow model?
#Save a model using `model.save` and load it using `tf.keras.models.load_model`:
model.save('my_model.h5')  # Save model
from tensorflow.keras.models import load_model
loaded_model = load_model('my_model.h5')  # Load model
