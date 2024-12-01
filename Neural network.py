"""
Question : What is deep learning, and how is it connected to artificial intelligence?
Deep learning is a branch of machine learning foc# Used on neural networks with multiple layers, enabling machines to perform tasks like image recognition, NLP, and autonomous driving. It's a foundational technology in AI.
Example: Using CNNs for detecting objects in images.

Question : What is a neural network, and what are the different types of neural networks?
A neural network mimics the brain's structure using interconnected nodes (neurons). Types include:
Question : CNNs for image processing, RNNs for sequential data, DNNs for general tasks, GANs for generative tasks.
Example: CNN for facial recognition, RNN for stock price prediction.

Question : What is the mathematical structure of a neural network?
A neural network has layers where inputs are multiplied by weights, added to biases, and passed through activation functions.
Example: z = Wx + b; output = f(z), where f is the activation function.

Question : What is an activation function, and why is it essential in neural networks?
An activation function adds non-linearity to the model, enabling it to learn complex patterns rather than just linear relationships.
Example: ReLU(x) = max(0, x), widely # Used in hidden layers.

Question : Could you list some common activation functions # Used in neural networks?
- Sigmoid: Maps values to [0,1], # Used for probabilities.
- Tanh: Maps values to [-1, 1], centered at 0.
- ReLU: Removes negative values (efficient for deep networks).
- Softmax: Outputs probabilities for multi-class classification.
Example: ReLU(x) = max(0, x); Softmax for multi-class problems.

Question : What is a multilayer neural network?
A neural network with multiple hidden layers that can learn hierarchical features.
Example: A deep CNN with multiple convolutional and pooling layers for image analysis.

Question : What is a loss function, and why is it crucial for neural network training?
A loss function quantifies the error between predicted and actual outputs, guiding the optimizer to minimize this error.
Example: MSE for regression problems; Cross-Entropy for classification.

Question : What are some common types of loss functions?
- Mean Squared Error (MSE): For regression tasks.
- Binary Cross-Entropy: For binary classification.
- Categorical Cross-Entropy: For multi-class classification.
Example: MSE = mean((y_pred - y_actual)^2).

Question : How does a neural network learn?
A neural network learns by updating weights to minimize the loss using backpropagation and optimizers like gradient descent.
Example: Gradient update: w = w - lr * grad(loss).

Question : What is an optimizer in neural networks, and why is it necessary?
Optimizers adjust weights efficiently to minimize the loss. They control the speed and quality of convergence.
Example: Adam combines momentum and adaptive learning rates.

Question : Could you briefly describe some common optimizers?
- SGD: Simple gradient descent.
- Adam: Combines momentum and adaptive learning.
- RMSprop: Adjusts learning rates for each parameter.
Example: Adam adapts learning rates dynamically for faster convergence.

Question : Can you explain forward and backward propagation in a neural network?
- Forward Propagation: # Computes outputs by passing inputs through layers.
- Backward Propagation: Calculates gradients and updates weights to minimize loss.
Example: Forward: input -> hidden layers -> output. Backward: loss -> gradients -> weight updates.

Question : What is weight initialization, and how does it impact training?
Weight initialization sets starting values for weights. Good initialization ensures stable gradients and faster convergence.
Example: Xavier initialization balances variance across layers.

Question : What is the vanishing gradient problem in deep learning?
In deep networks, gradients diminish as they propagate backward, slowing training.
Example: Using ReLU instead of Sigmoid mitigates this issue.

Question : What is the exploding gradient problem?
Gradients grow excessively in deep networks, causing instability or divergence.
Example: Gradient clipping prevents weights from exploding.

"""

# Question :How do you create a simple perceptron for basic binary classification?
# A perceptron has one layer with a binary output. In Keras:
from keras.models import Sequential
from keras.layers import Dense
model = Sequential([Dense(1, activation='sigmoid', input_dim=2)])  # Question :1 output, sigmoid for binary classification
model.compile(optimizer='sgd', loss='binary_crossentropy')

# Question :How can you build a neural network with one hidden layer using Keras?
# Use `Sequential` with two layers:
model = Sequential([
    Dense(16, activation='relu', input_dim=4),  # Question :Hidden layer with 16 neurons
    Dense(1, activation='sigmoid')  # Question :Output layer
])
model.compile(optimizer='adam', loss='binary_crossentropy')

# Question :How do you initialize weights using the Xavier (Glorot) initialization method in Keras?
# Use `kernel_initializer='glorot_uniform'` when defining layers:
model.add(Dense(16, activation='relu', kernel_initializer='glorot_uniform'))

# Question :How can you apply different activation functions in a neural network in Keras?
# Specify the activation function in each layer:
Dense(32, activation='relu')  # Question :ReLU
Dense(1, activation='sigmoid')  # Question :Sigmoid for binary output

# Question :How do you add dropout to a neural network model to prevent overfitting?
# Use `Dropout` layer:
from keras.layers import Dropout
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # Question :Drops 50% of neurons randomly

# Question :How do you manually implement forward propagation in a simple neural network?
# Compute outputs layer by layer:
import numpy as np
X = np.array([[0.5, 0.2]])  # Question :Example input
W = np.random.rand(2, 1)  # Question :Weights
b = np.random.rand(1)  # Question :Bias
z = np.dot(X, W) + b  # Question :Linear combination
a = 1 / (1 + np.exp(-z))  # Question :Sigmoid activation

# Question :How do you add batch normalization to a neural network model in Keras?
# Use `BatchNormalization` layer:
from keras.layers import BatchNormalization
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())

# Question :How can you visualize the training process with accuracy and loss curves?
# Use the history object returned by `model.fit`:
import matplotlib.pyplot as plt
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()

# Question :How can you # Use gradient clipping in Keras to control the gradient size and prevent exploding gradients?
# Specify `clipvalue` or `clipnorm` in optimizer:
from keras.optimizers import Adam
optimizer = Adam(clipvalue=1.0)  # Question :Clip gradients to range [-1, 1]
model.compile(optimizer=optimizer, loss='mse')

# Question :How can you create a custom loss function in Keras?
# Define a Python function and pass it to `model.compile`:
import keras.backend as K
def custom_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))
model.compile(optimizer='adam', loss=custom_loss)

# Question :How can you visualize the structure of a neural network model in Keras?
# Use `plot_model` from `keras.utils`:
from keras.utils import plot_model
plot_model(model, to_file='model_structure.png', show_shapes=True)

