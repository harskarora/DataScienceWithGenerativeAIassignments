"""
1. What is BERT and how does it work?

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained NLP model that uses the Transformer architecture.
It learns bidirectional context by simultaneously looking at both left and right contexts of a word during training.
Example: BERT is used for tasks like sentiment analysis and question answering.


2. What are the main advantages of using the attention mechanism in neural networks?

Advantages:
- Focuses on relevant parts of input sequences, improving performance on sequence tasks.
- Handles long-range dependencies better than RNNs or LSTMs.
Example: Attention helps in machine translation by aligning source and target words.


3. How does the self-attention mechanism differ from traditional attention mechanisms?

Self-attention computes relationships among words in a single input sequence, unlike traditional attention that focuses between two sequences (e.g., source and target).
Example: Transformers use self-attention to encode sentence representations efficiently.


4. What is the role of the decoder in a Seq2Seq model?

The decoder generates the output sequence (e.g., translations) using the encoder's context and previous outputs.
Example: In translation, the decoder produces "Bonjour" from "Hello."


5. What is the difference between GPT-2 and BERT models?

- GPT-2 is an autoregressive model focused on text generation.
- BERT is a bidirectional model designed for understanding context.
Example: GPT-2 excels at completing sentences, while BERT handles classification.


6. Why is the Transformer model considered more efficient than RNNs and LSTMs?

Transformers process sequences in parallel using attention mechanisms, avoiding the sequential nature of RNNs.
Example: Transformers reduce training time for tasks like summarization.


7. Explain how the attention mechanism works in a Transformer model.

Attention computes weights for input tokens based on relevance using key, query, and value vectors.
Example: "The cat sat" assigns high weights to related words like "cat" and "sat."


8. What is the difference between an encoder and a decoder in a Seq2Seq model?

The encoder processes the input into a context vector, and the decoder generates outputs based on this vector.
Example: Encoder extracts features; decoder generates translations.


9. What is the primary purpose of using the self-attention mechanism in transformers?

Self-attention enables models to focus on important parts of input sequences, improving context understanding.
Example: Resolving word ambiguity in sentences like "He saw a bank."


10. How does the GPT-2 model generate text?

GPT-2 generates text by predicting the next token in a sequence based on prior tokens.
Example: Input: "The weather today is," Output: "sunny and warm."


11. Explain the concept of “fine-tuning” in BERT.

Fine-tuning adapts a pre-trained BERT model to specific tasks by training on labeled task-specific data.
Example: Fine-tuning BERT for sentiment classification.


12. What is the main difference between the encoder-decoder architecture and a simple neural network?

Encoder-decoder handles sequence-to-sequence tasks with two components, while simple neural networks typically process fixed-size inputs/outputs.
Example: Encoder-decoder translates text; simple NN predicts numeric outputs.


13. How does the attention mechanism handle long-range dependencies in sequences?

Attention computes global relationships, avoiding distance-based limitations in RNNs.
Example: Recognizing "Paris" and "France" connection in a long sentence.


14. What is the core principle behind the Transformer architecture?

The core principle is self-attention and parallel processing, allowing efficient handling of sequences.
Example: Transformers excel in text summarization tasks.


15. What is the role of the "position encoding" in a Transformer model?

Position encoding adds sequence order information since Transformers process tokens in parallel.
Example: Encodes "first" and "last" token positions.


16. How do Transformers use multiple layers of attention?

Each layer refines learned representations, improving context understanding.
Example: Early layers capture syntax; deeper layers capture semantics.


17. What does it mean when a model is described as “autoregressive” like GPT-2?

Autoregressive models generate outputs sequentially, conditioning on prior tokens.
Example: GPT-2 predicts one word at a time for text generation.


18. How does BERT's bidirectional training improve its performance?

Bidirectional training captures full sentence context, unlike unidirectional models.
Example: Understanding "bank" as "financial" or "river" based on context.


19. What are the advantages of using the Transformer over RNN-based models in NLP?

Advantages:
- Parallelism in training.
- Handles long sequences effectively.
- Higher scalability.
Example: Transformers outperform RNNs in translation tasks.


20. What is the attention mechanism’s impact on the performance of models like BERT and GPT-2?

Attention improves context understanding and efficiency, enabling superior performance on tasks like classification and generation.
Example: GPT-2 generates coherent paragraphs due to attention mechanisms.

"""

# 1. How to implement a simple text classification model using LSTM in Keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import numpy as np

# Sample data
texts = ["I love NLP", "Deep learning is fascinating", "I dislike slow models"]
labels = [1, 1, 0]  # 1: Positive, 0: Negative

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=10)

# Model
model = Sequential([
    Embedding(input_dim=1000, output_dim=32, input_length=10),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Dummy training
labels = np.array(labels)
model.fit(data, labels, epochs=3, batch_size=2)

# 2. How to generate sequences of text using a Recurrent Neural Network (RNN)
from keras.layers import SimpleRNN
model = Sequential([
    Embedding(input_dim=1000, output_dim=32, input_length=10),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])
model.summary()

# 3. How to perform sentiment analysis using a simple CNN model
from keras.layers import Conv1D, GlobalMaxPooling1D
model = Sequential([
    Embedding(input_dim=1000, output_dim=32, input_length=10),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(1, activation='sigmoid')
])
model.summary()

# 4. How to perform Named Entity Recognition (NER) using spaCy
import spacy
nlp = spacy.load("en_core_web_sm")
text = "Apple is looking to buy a startup in London."
doc = nlp(text)
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

# 5. How to implement a simple Seq2Seq model for machine translation using LSTM in Keras
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

input_seq = Input(shape=(10, 100))  # Input shape: (sequence_length, feature_dim)
encoder = LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder(input_seq)
encoder_states = [state_h, state_c]

decoder = LSTM(64, return_sequences=True, return_state=False)
decoder_outputs = decoder(RepeatVector(10)(encoder_outputs))
output = Dense(100, activation='softmax')(decoder_outputs)

seq2seq_model = Model(input_seq, output)
seq2seq_model.summary()

# 6. How to generate text using a pre-trained transformer model (GPT-2)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# 7. How to apply data augmentation for text in NLP
from nlpaug.augmenter.word import SynonymAug
text = "Natural Language Processing is amazing."
augmenter = SynonymAug()
augmented_text = augmenter.augment(text)
print(f"Original: {text}\nAugmented: {augmented_text}")

# 8. How can you add an Attention Mechanism to a Seq2Seq model?
from keras.layers import Dot, Activation, Concatenate
def attention_layer(inputs):
    # inputs: [encoder_outputs, decoder_outputs]
    encoder_outputs, decoder_outputs = inputs

    # Calculate attention scores
    scores = Dot(axes=[2, 2])([decoder_outputs, encoder_outputs])
    attention_weights = Activation('softmax')(scores)

    # Context vector
    context = Dot(axes=[2, 1])([attention_weights, encoder_outputs])

    # Combine context with decoder output
    combined = Concatenate()([context, decoder_outputs])
    return combined

# Example usage with Seq2Seq model:
attention_outputs = attention_layer([encoder_outputs, decoder_outputs])
