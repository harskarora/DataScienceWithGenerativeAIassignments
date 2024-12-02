"""
# What does GAN stand for, and what is its main purpose?
Answer: GAN stands for Generative Adversarial Network. Its main purpose is to generate realistic data (e.g., images, audio) by training two neural networks in opposition.

# Explain the concept of the "discriminator" in GANs.
Answer: The discriminator is a neural network in a GAN that distinguishes between real and generated (fake) data, helping improve the generator's output quality.

# How does a GAN work?
Answer: A GAN consists of a generator and a discriminator. The generator creates fake data, while the discriminator evaluates it. Both improve iteratively through adversarial training.

# What is the generator's role in a GAN?
Answer: The generator's role is to create data that mimics the real data distribution, fooling the discriminator into classifying it as real.

# What is the loss function used in the training of GANs?
Answer: GANs typically use a binary cross-entropy loss to evaluate how well the generator and discriminator perform against each other.

# What is the difference between a WGAN and a traditional GAN?
Answer: WGAN (Wasserstein GAN) replaces the binary cross-entropy loss with the Wasserstein loss, improving training stability and addressing mode collapse.

# How does the training of the generator differ from that of the discriminator?
Answer: The generator is trained to maximize the discriminator's error, while the discriminator is trained to correctly classify real and fake data.

# What is a DCGAN, and how is it different from a traditional GAN?
Answer: DCGAN (Deep Convolutional GAN) uses convolutional layers, making it more effective for image generation than traditional GANs with dense layers.

# Explain the concept of "controllable generation" in the context of GANs.
Answer: Controllable generation involves guiding GANs to produce outputs with specific attributes (e.g., a smiling face) by modifying latent space representations.

# What is the primary goal of training a GAN?
Answer: The primary goal is to train the generator to produce data indistinguishable from real data as judged by the discriminator.

# What are the limitations of GANs?
Answer: Limitations include mode collapse, unstable training, and the need for large datasets and computational resources.

# What are StyleGANs, and what makes them unique?
Answer: StyleGANs generate high-quality images by introducing style-based latent space manipulation, enabling fine-grained control over image attributes.

# What is the role of noise in a GAN?
Answer: Noise serves as input to the generator, which transforms it into realistic outputs by learning the data distribution.

# Describe the architecture of a typical GAN.
Answer: A typical GAN consists of a generator (producing fake data) and a discriminator (evaluating real vs. fake) trained in a competitive framework.

# How does the loss function in a WGAN improve training stability?
Answer: WGAN's Wasserstein loss ensures smoother gradients and resolves vanishing gradient issues, making training more stable.

# What challenges do GANs face during training, and how can they be addressed?
Answer: Challenges include mode collapse, instability, and vanishing gradients. Solutions include WGANs, batch normalization, and learning rate adjustments.

# How does DCGAN help improve image generation in GANs?
Answer: DCGAN uses convolutional layers, batch normalization, and ReLU/Leaky ReLU activations to enhance image quality and stability during training.

# What are the key differences between a traditional GAN and a StyleGAN?
Answer: StyleGAN introduces style-based synthesis and disentangling latent spaces, enabling more controllable and realistic image generation.

# How does the discriminator decide whether an image is real or fake in a GAN?
Answer: The discriminator assigns a probability (real vs. fake) based on features extracted from the input image.

# What is the main advantage of using GANs in image generation?
Answer: GANs produce highly realistic and diverse images without explicitly modeling the data distribution.

# How can GANs be used in real-world applications?
Answer: Applications include image generation, super-resolution, data augmentation, medical imaging, and art creation.

# What is Mode Collapse in GANs, and how can it be prevented?
Answer: Mode collapse occurs when the generator produces limited outputs. It can be mitigated using WGANs, diversity-promoting losses, or architectural improvements.


"""


import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import numpy as np

# 1. Implement a simple GAN architecture to generate random images (like noise or basic shapes) using TensorFlow/Keras.
def create_simple_gan(latent_dim=100, img_shape=(28, 28, 1)):
    # Generator
    generator = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.Dense(np.prod(img_shape), activation='sigmoid'),
        layers.Reshape(img_shape)
    ])

    # Discriminator
    discriminator = tf.keras.Sequential([
        layers.Flatten(input_shape=img_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return generator, discriminator

# 2. Implement the discriminator for a GAN with an image input of shape (28, 28).
def build_discriminator(img_shape=(28, 28, 1)):
    discriminator = tf.keras.Sequential([
        layers.Flatten(input_shape=img_shape),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return discriminator

# 3. Train the generator to produce simple digits (using noise as input) and plot the generated images.
def train_simple_gan(generator, discriminator, epochs=1000, batch_size=32, latent_dim=100):
    half_batch = batch_size // 2
    img_shape = generator.output_shape[1:]

    for epoch in range(epochs):
        # Train Discriminator
        real_imgs = np.random.rand(half_batch, *img_shape)
        fake_imgs = generator.predict(np.random.normal(0, 1, (half_batch, latent_dim)))

        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((half_batch, 1)))

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        g_loss = discriminator.train_on_batch(generator.predict(noise), valid_y)

    return generator

# Plot generated images
def plot_generated_images(generator, latent_dim=100, n_images=5):
    noise = np.random.normal(0, 1, (n_images, latent_dim))
    gen_images = generator.predict(noise)
    for i in range(n_images):
        plt.imshow(gen_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.show()

# 4. Implement WGAN by modifying the loss function in the GAN.
def wasserstein_loss(y_true, y_pred):
    return -tf.reduce_mean(y_true * y_pred)

# 5. Use a trained generator to generate a batch of fake images and display them.
def generate_fake_images(generator, n_images=10, latent_dim=100):
    noise = np.random.normal(0, 1, (n_images, latent_dim))
    fake_images = generator.predict(noise)
    for img in fake_images:
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.show()

# 6. Create a StyleGAN-inspired architecture that outputs high-resolution images.
# Simplified example
def create_stylegan_generator(latent_dim=100, img_shape=(64, 64, 3)):
    generator = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.Reshape((4, 4, 8)),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"),
        layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", activation="relu"),
        layers.Conv2DTranspose(img_shape[-1], kernel_size=4, strides=2, padding="same", activation="tanh"),
    ])
    return generator

# 7. Implement the Wasserstein loss function for GAN training.
# (Already implemented above as wasserstein_loss function.)

# 8. Write a function to modify the discriminator to include a dropout layer with a rate of 0.4 and print the configurations.
def add_dropout_to_discriminator(img_shape=(28, 28, 1), dropout_rate=0.4):
    discriminator = tf.keras.Sequential([
        layers.Flatten(input_shape=img_shape),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    print(discriminator.summary())
    return discriminator

# Example usage
latent_dim = 100
generator, discriminator = create_simple_gan(latent_dim)
train_simple_gan(generator, discriminator, epochs=10)
plot_generated_images(generator, latent_dim)

new_discriminator = add_dropout_to_discriminator()
