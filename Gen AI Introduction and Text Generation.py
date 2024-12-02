"""
1. What is Generative AI?

Generative AI refers to a type of AI that creates new content, such as images, text, music, or videos, based on patterns learned from data.
Example: GPT models generating human-like text.


2. How is Generative AI different from traditional AI?

Generative AI creates new content, while traditional AI focuses on pattern recognition and decision-making based on existing data.
Example: Traditional AI classifies images; Generative AI creates new images.


3. Name two applications of Generative AI in the industry.

1. Text Generation: GPT models are used for content generation in marketing and chatbots.
2. Image Synthesis: GANs are used to create realistic images and artwork.


4. What are some challenges associated with Generative AI?

Challenges include ensuring generated content is meaningful, avoiding bias, and dealing with computational resource demands.
Example: Ensuring AI-generated text doesn't propagate harmful stereotypes.


5. Why is Generative AI important for modern applications?

Generative AI enables the creation of new, diverse content at scale, which is valuable for personalized marketing, entertainment, and automation.
Example: Personalized ads generated based on customer preferences.


6. What is probabilistic modeling in the context of Generative AI?

Probabilistic modeling refers to using probability distributions to model uncertainty in data and generate new instances that are likely based on learned patterns.
Example: GANs and VAEs use probabilistic models to generate realistic images and text.


7. Define a generative model.

A generative model learns the underlying distribution of data and can generate new data that mimics this distribution.
Example: Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN) are generative models.


8. Explain how an n-gram model works in text generation.

An n-gram model predicts the next word in a sequence based on the previous n-1 words. It works by estimating the probability of the next word.
Example: "I love AI" -> 2-gram model predicts "AI" given "I love".


9. What are the limitations of n-gram models?

N-gram models are limited by their inability to capture long-range dependencies and the curse of dimensionality with larger n-values.
Example: A bigram model might struggle to generate coherent sentences.


10. How can you improve the performance of an n-gram model?

Increasing the value of n (e.g., using a trigram model) or incorporating smoothing techniques like Laplace smoothing can improve performance.
Example: A trigram model captures better context than a bigram.


11. What is the Markov assumption, and how does it apply to text generation?

The Markov assumption states that the future state depends only on the current state, not on past states. In text generation, this means predicting a word based only on the previous word(s).
Example: In text generation, the next word depends only on the immediate previous word.


12. Why are probabilistic models important in generative AI?

Probabilistic models capture uncertainty in data and provide a way to generate new, diverse, and realistic outputs.
Example: VAEs generate variations of data by sampling from a probability distribution.


13. What is an autoencoder?

An autoencoder is a neural network used for unsupervised learning, which compresses input data into a lower-dimensional space (encoding) and then reconstructs the data (decoding).
Example: An autoencoder might reduce the dimensions of an image while preserving important features.


14. How does a VAE differ from a standard autoencoder?

A VAE introduces a probabilistic approach, learning a distribution over the latent space instead of a fixed vector.
Example: A VAE can generate new data by sampling from the latent space, unlike a regular autoencoder.


15. Why are VAEs useful in generative modeling?

VAEs are useful because they can model complex data distributions and generate new data by sampling from the learned latent space.
Example: VAEs are used for generating images and text.


16. What role does the decoder play in an autoencoder?

The decoder in an autoencoder reconstructs the input data from the lower-dimensional encoding.
Example: In an image autoencoder, the decoder reconstructs the image after encoding it.


17. How does the latent space affect text generation in a VAE?

The latent space in a VAE represents compressed, learned features of the data. Sampling from this space generates new data that mimics the distribution of the original data.
Example: In text generation, the latent space influences the diversity and coherence of generated text.


18. What is the purpose of the Kullback-Leibler (KL) divergence term in VAEs?

KL divergence measures how much the learned distribution deviates from the desired prior distribution. It ensures the latent space is structured and prevents overfitting.
Example: In a VAE, KL divergence helps regularize the model.


19. How can you prevent overfitting in a VAE?

Overfitting in VAEs can be prevented by using regularization techniques like KL divergence, adding dropout layers, and using early stopping during training.
Example: Dropout in the encoder and decoder layers helps regularize the model.


20. What is a transformer model?

A transformer is a deep learning model designed to handle sequences, using self-attention mechanisms instead of recurrent layers to process data in parallel.
Example: Transformers power models like GPT and BERT.


21. Explain the purpose of self-attention in transformers.

Self-attention enables the model to weigh the importance of different tokens in a sequence, allowing it to capture long-range dependencies efficiently.
Example: In translation, self-attention allows the model to align words across languages.


22. How does a GPT model generate text?

GPT models generate text by predicting the next word in a sequence based on prior words, using an autoregressive approach.
Example: "The weather today is" -> GPT generates "sunny."


23. Explain why VAEs are commonly used for unsupervised learning tasks.

VAEs are useful for unsupervised learning because they model data distributions and learn useful representations of the data without requiring labels.
Example: VAEs are used for anomaly detection and data generation.


24. What are the key differences between a GPT model and an RNN?

GPT models use a transformer architecture with self-attention, which processes all tokens simultaneously, while RNNs process tokens sequentially.
Example: GPT can handle long-range dependencies better than RNNs.


25. How does fine-tuning improve a pre-trained GPT model?

Fine-tuning adapts a pre-trained GPT model to a specific task or domain by training it on task-specific labeled data.
Example: Fine-tuning GPT on a sentiment analysis dataset improves its performance for that task.


26. What is zero-shot learning in the context of GPT models?

Zero-shot learning refers to the ability of a model to perform a task without task-specific training by leveraging general knowledge learned during pre-training.
Example: GPT can answer questions or perform classification without specific task training.


27. Describe how prompt engineering can impact GPT model performance.

Prompt engineering involves crafting inputs to guide the model toward desired outputs. It can significantly affect the quality and relevance of the model's responses.
Example: A well-structured prompt can improve the coherence of GPT's text generation.


28. Why are large datasets essential for training GPT models?

Large datasets enable GPT models to learn diverse language patterns and contexts, making them more capable of handling a wide range of tasks.
Example: GPT's ability to generate coherent text is largely due to the vast data it is trained on.


29. What are potential ethical concerns with GPT models?

Ethical concerns include bias in generated text, misinformation propagation, and the potential for generating harmful content.
Example: GPT models may inadvertently generate biased or offensive language.


30. How does the attention mechanism contribute to GPT’s ability to handle long-range dependencies?

The attention mechanism in GPT allows the model to focus on relevant tokens from any part of the sequence, making it capable of handling long-range dependencies efficiently.
Example: GPT can connect "long" and "range" even if they appear far apart in a sentence.


31. What are some limitations of GPT models for real-world applications?

Limitations include reliance on large amounts of data, the inability to verify factual correctness, and the generation of biased content.
Example: GPT may generate plausible-sounding but factually incorrect statements.


32. How can GPT models be adapted for domain-specific text generation?

GPT models can be fine-tuned on domain-specific data to improve their relevance and accuracy for particular industries or topics.
Example: Fine-tuning GPT on medical literature makes it suitable for medical text generation.


33. What are some common metrics for evaluating text generation quality?

Common metrics include BLEU, ROUGE, and perplexity, which evaluate factors like fluency, relevance, and coherence of the generated text.
Example: BLEU measures the quality of machine-generated text by comparing it to reference texts.


34. Explain the difference between deterministic and probabilistic text generation.

Deterministic generation produces the same output for the same input every time, while probabilistic generation introduces variability, allowing for diverse outputs.
Example: GPT generates different outputs for the same prompt due to probabilistic sampling.


35. How does beam search improve text generation in language models?

Beam search explores multiple possible outputs by maintaining the top-k most probable sequences, improving the diversity and quality of generated text.
Example: Beam search helps GPT generate more fluent and coherent sentences.


"""

# 1. Generate a random sentence using probabilistic modeling (Markov Chain)
import random

# Example sentence: "The cat is on the mat"
sentence = "The cat is on the mat"
words = sentence.split()

# Build a simple Markov Chain model
def build_markov_chain(words):
    markov_chain = {}
    for i in range(len(words) - 1):
        if words[i] not in markov_chain:
            markov_chain[words[i]] = []
        markov_chain[words[i]].append(words[i+1])
    return markov_chain

# Function to generate a sentence using the Markov Chain
def generate_sentence(markov_chain, start_word, length=10):
    current_word = start_word
    result = [current_word]
    for _ in range(length - 1):
        next_word = random.choice(markov_chain.get(current_word, [""]))
        result.append(next_word)
        current_word = next_word
    return ' '.join(result)

markov_chain = build_markov_chain(words)
random_sentence = generate_sentence(markov_chain, "The")
print(random_sentence)


# 2. Build a simple Autoencoder model using Keras to learn a compressed representation of a given sentence
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np

# Example dataset: Simple sentences represented as one-hot encoded vectors
sentences = ["The cat is on the mat", "The dog barks loudly", "The quick brown fox jumps over the lazy dog"]
max_sentence_length = max(len(sentence.split()) for sentence in sentences)

# Preprocessing: One-hot encoding
def sentence_to_one_hot(sentence, max_length):
    words = sentence.split()
    one_hot = np.zeros((max_length, 1))
    for i, word in enumerate(words):
        one_hot[i, 0] = hash(word) % 1000  # Simple hashing for illustration
    return one_hot.T

X_train = np.array([sentence_to_one_hot(sentence, max_sentence_length) for sentence in sentences])

# Build Autoencoder
input_layer = Input(shape=(max_sentence_length, 1))
encoded = Dense(64, activation='relu')(input_layer)
decoded = Dense(max_sentence_length, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_train, X_train, epochs=10, batch_size=1)

# 3. Use the Hugging Face transformers library to fine-tune a pre-trained GPT-2 model on custom text data
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Custom dataset (for illustration)
text = "The cat sat on the mat. The dog barks loudly."
train_encodings = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=True)

# Set up fine-tuning
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings
)

trainer.train()

# 4. Implement a text generation model using a simple Recurrent Neural Network (RNN) in Keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding

# Prepare data (example)
data = "The cat sat on the mat. The dog barks loudly."
tokens = data.split()
vocab_size = len(set(tokens))

# Define the RNN model
model = Sequential([
    Embedding(vocab_size, 64),
    SimpleRNN(128, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model (dummy training for example purposes)
# train_data = prepare_data(data)
# model.fit(train_data, epochs=10)

# 5. Program to generate a sequence of text using an LSTM-based model in TensorFlow
import tensorflow as tf

# Prepare data for LSTM
data = ["The cat is on the mat.", "The dog barks loudly.", "The fox jumps over the lazy dog."]
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(len(tokenizer.word_index)+1, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# 6. Program that uses GPT-2 from Hugging Face to generate a story based on a custom prompt
input_text = "Once upon a time"
generated = tokenizer.encode(input_text, return_tensors='pt')

output = model.generate(generated, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# 7. Implement a simple text generation model using a GRU-based architecture in Keras
model_gru = Sequential([
    Embedding(vocab_size, 64),
    GRU(128),
    Dense(vocab_size, activation='softmax')
])

model_gru.compile(optimizer='adam', loss='categorical_crossentropy')

# 8. GPT-2-based text generation with beam search decoding
output = model.generate(generated, num_beams=5, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# 9. GPT-2 with custom temperature setting for diversity in output text
output_temp = model.generate(generated, max_length=50, temperature=0.7)
print(tokenizer.decode(output_temp[0], skip_special_tokens=True))

# 10. Temperature sampling with GPT-2
output_temp_sample = model.generate(generated, max_length=50, temperature=0.9)
print(tokenizer.decode(output_temp_sample[0], skip_special_tokens=True))

# 11. Simple LSTM-based text generation model from scratch using Keras
model_lstm = Sequential([
    Embedding(vocab_size, 64),
    LSTM(128),
    Dense(vocab_size, activation='softmax')
])

model_lstm.compile(optimizer='adam', loss='categorical_crossentropy')
model_lstm.fit(train_data, epochs=10)

# 12. Implement text generation using a simple custom attention-based architecture
from keras.layers import Attention, Add, Dense

inputs = Input(shape=(None, 64))
x = Attention()([inputs, inputs])
x = Add()([x, inputs])
x = Dense(64)(x)
output = Dense(vocab_size, activation='softmax')(x)

model_attention = Model(inputs, output)
model_attention.compile(optimizer='adam', loss='categorical_crossentropy')

# Example usage for training and text generation
model_attention.fit(train_data, epochs=10)
