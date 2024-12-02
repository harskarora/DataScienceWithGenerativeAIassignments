"""
 1. What is NLTK?
Answer: NLTK (Natural Language Toolkit) is a Python library for NLP tasks like tokenization, stemming, lemmatization, and parsing.
import nltk
print("NLTK is a library for performing NLP tasks.")

 2. What is SpaCy and how does it differ from NLTK?
Answer: SpaCy is an NLP library focused on industrial use cases, offering faster processing and pretrained models.
import spacy
print("SpaCy is designed for production, while NLTK is better for academic purposes.")

 3. What is the purpose of TextBlob in NLP?
Answer: TextBlob simplifies text processing with easy-to-use APIs for tokenization, sentiment analysis, and NER.
from textblob import TextBlob
blob = TextBlob("TextBlob makes text processing simple.")
print(blob.sentiment)

 4. What is Stanford NLP?
Answer: Stanford NLP provides state-of-the-art tools for NLP tasks, including dependency parsing and NER.
print("Stanford NLP offers Java-based NLP solutions.")

 5. Explain what Recurrent Neural Networks (RNN) are.
Answer: RNNs are neural networks designed for sequential data processing, with connections forming directed cycles.
print("RNNs process sequences by maintaining hidden states.")

 6. What is the main advantage of using LSTM over RNN?
Answer: LSTMs address vanishing gradient issues in RNNs by using gates to regulate information flow.
print("LSTMs solve long-term dependency issues in RNNs.")

 7. What are Bi-directional LSTMs, and how do they differ from standard LSTMs?
Answer: Bi-LSTMs process sequences in both forward and backward directions for better context understanding.
print("Bi-LSTMs combine future and past context.")

 8. What is the purpose of a Stacked LSTM?
Answer: Stacked LSTMs increase model capacity by stacking multiple LSTM layers.
print("Stacked LSTMs improve feature representation.")

 9. How does a GRU (Gated Recurrent Unit) differ from an LSTM?
Answer: GRUs simplify LSTMs by combining the forget and input gates into a single update gate.
print("GRUs are computationally efficient compared to LSTMs.")

 10. What are the key features of NLTK's tokenization process?
Answer: NLTK supports both word and sentence tokenization with language-specific models.
sentence = "Tokenization is the first step in text preprocessing."
tokens = nltk.word_tokenize(sentence)
print(f"Tokens: {tokens}")

 11. How do you perform named entity recognition (NER) using SpaCy?
doc = nlp("Apple is looking at buying a U.K. startup for $1 billion.")
entities = [(ent.text, ent.label_) for ent in doc.ents]
print(f"Named Entities: {entities}")

 12. What is Word2Vec and how does it represent words?
Answer: Word2Vec creates vector representations of words based on their context in a corpus.
from gensim.models import Word2Vec
sentences = [["hello", "world"], ["machine", "learning", "is", "fun"]]
model = Word2Vec(sentences, vector_size=10, min_count=1)
print(f"Word2Vec vector for 'hello': {model.wv['hello']}")

 13. Explain the difference between Bag of Words (BoW) and Word2Vec.
Answer: BoW represents text as frequency counts, while Word2Vec uses dense vector embeddings.
print("BoW ignores semantics, Word2Vec captures contextual relationships.")

 14. How does TextBlob handle sentiment analysis?
text = TextBlob("TextBlob is amazing.")
print(f"Sentiment: {text.sentiment}")

 15. How would you implement text preprocessing using NLTK?
tokens = nltk.word_tokenize("This is an example text.")
stop_words = set(nltk.corpus.stopwords.words("english"))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(f"Preprocessed Tokens: {filtered_tokens}")

 16. How do you train a custom NER model using SpaCy?
Answer: Create training data, add a blank entity recognizer, and train the model using `nlp.update()`.
print("Train a SpaCy NER model with custom labeled data.")

 17. What is the role of the attention mechanism in LSTMs and GRUs?
Answer: Attention focuses on relevant parts of the input sequence, improving sequence-to-sequence tasks.
print("Attention improves context retention in LSTMs and GRUs.")

 18. What is the difference between tokenization and lemmatization in NLP?
Answer: Tokenization splits text into units; lemmatization reduces words to their base form.
print("Tokenization creates tokens, lemmatization normalizes them.")

 19. How do you perform text normalization in NLP?
normalized = sentence.lower().translate(str.maketrans('', '', string.punctuation))
print(f"Normalized Text: {normalized}")

 20. What is the purpose of frequency distribution in NLP?
Answer: Frequency distribution identifies common terms in text for analysis.
from collections import Counter
text = "apple banana apple orange banana apple"
freq_dist = Counter(text.split())
print(f"Frequency Distribution: {freq_dist}")

 21. What are co-occurrence vectors in NLP?
Answer: Represent relationships based on how often words appear together in context.
print("Co-occurrence vectors encode word relationships.")

 22. How is Word2Vec used to find the relationship between words?
 Example: Similar words to 'machine' in a trained Word2Vec model.
print(f"Similar words to 'machine': {model.wv.most_similar('machine')}")

 23. How does a Bi-LSTM improve NLP tasks compared to a regular LSTM?
Answer: Bi-LSTMs consider context from both directions.
print("Bi-LSTMs enhance context comprehension.")

 24. What is the difference between a GRU and an LSTM in terms of gate structures?
Answer: GRUs have fewer gates, making them computationally lighter.
print("GRUs combine gates for efficiency.")

 25. How does Stanford NLP’s dependency parsing work?
Answer: Analyzes grammatical relationships between words in a sentence.
print("Stanford NLP uses tree structures for dependency parsing.")

 26. How does tokenization affect downstream NLP tasks?
Answer: Tokenization determines how text is split, affecting model inputs.
print("Tokenization quality impacts all downstream tasks.")

 27. What are some common applications of NLP?
Answer: Machine translation, sentiment analysis, chatbots, and text summarization.
print("Common NLP applications: Translation, chatbots, etc.")

 28. What are stopwords and why are they removed in NLP?
Answer: Stopwords are common words with little semantic value, removed to reduce noise.
print("Removing stopwords focuses on meaningful terms.")

 29. How can you implement word embeddings using Word2Vec in Python?
Answer: Use Gensim's Word2Vec to train or load pretrained embeddings.
print(f"Word2Vec Embedding: {model.wv['learning']}")

 30. How does SpaCy handle lemmatization?
Answer: Uses linguistic rules and lexicons for accurate lemmatization.
doc = nlp("The leaves are falling.")
print([token.lemma_ for token in doc])

 31. What is the significance of RNNs in NLP tasks?
Answer: RNNs process sequential data, capturing dependencies over time.
print("RNNs are essential for sequence-based tasks.")

 32. How does word embedding improve the performance of NLP models?
Answer: Embeddings capture semantic relationships, boosting accuracy.
print("Embeddings add semantic understanding to models.")

 33. How does a Stacked LSTM differ from a single LSTM?
Answer: Stacked LSTMs have multiple layers, increasing model depth.
print("Stacked LSTMs capture complex patterns.")

 34. What are the key differences between RNN, LSTM, and GRU?
Answer: RNNs struggle with long-term dependencies; LSTMs and GRUs address this with gating mechanisms.
print("RNNs suffer from vanishing gradients; LSTMs/GRUs resolve it.")

 35. Why is the attention mechanism important in sequence-to-sequence models?
Answer: Attention allows models to focus on relevant parts of input sequences for better predictions.
print("Attention enhances context sensitivity in seq-to-seq models.")

"""

# 1. How do you perform word tokenization using NLTK and plot a word frequency distribution?
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

text = "Natural Language Processing is fascinating. NLP is a branch of AI."
tokens = word_tokenize(text)
freq_dist = Counter(tokens)
plt.bar(freq_dist.keys(), freq_dist.values())
plt.title("Word Frequency Distribution")
plt.xticks(rotation=45)
plt.show()

# 2. How do you use SpaCy for dependency parsing of a sentence?
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("SpaCy makes dependency parsing easy.")
for token in doc:
    print(f"Word: {token.text}, Dependency: {token.dep_}, Head: {token.head.text}")

# 3. How do you use TextBlob for performing text classification based on polarity?
from textblob import TextBlob
blob = TextBlob("TextBlob is very helpful and easy to use.")
print(f"Polarity: {blob.sentiment.polarity}")

# 4. How do you extract named entities from a text using SpaCy?
text = "Google is planning to open an office in New York."
doc = nlp(text)
entities = [(ent.text, ent.label_) for ent in doc.ents]
print(f"Named Entities: {entities}")

# 5. How can you calculate TF-IDF scores for a given text using Scikit-learn?
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["This is a sample document.", "This document is another example."]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
print(f"TF-IDF Scores:\n{tfidf_matrix.toarray()}\nFeature Names: {vectorizer.get_feature_names_out()}")

# 6. How do you create a custom text classifier using NLTK's Naive Bayes classifier?
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

def format_data():
    return [(dict([(word, True) for word in movie_reviews.words(fileid)]), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

train_data = format_data()
classifier = NaiveBayesClassifier.train(train_data[:1500])
print(classifier.classify({'amazing': True, 'terrible': False}))

# 7. How do you use a pre-trained model from Hugging Face for text classification?
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face transformers.")
print(result)

# 8. How do you perform text summarization using Hugging Face transformers?
summarizer = pipeline("summarization")
summary = summarizer("Hugging Face transformers provide many state-of-the-art NLP models. They simplify the implementation of NLP tasks.")
print(summary)

# 9. How can you create a simple RNN for text classification using Keras?
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding

model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=100),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 10. How do you train a Bidirectional LSTM for text classification?
from keras.layers import Bidirectional, LSTM
model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=100),
    Bidirectional(LSTM(64)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 11. How do you implement GRU (Gated Recurrent Unit) for text classification?
from keras.layers import GRU
model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=100),
    GRU(32),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 12. How do you implement a text generation model using LSTM with Keras?
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM
import numpy as np

# Sample dataset for simplicity
texts = ["hello world", "hello keras", "hello NLP"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

model = Sequential([
    Embedding(input_dim=50, output_dim=10),
    LSTM(100, return_sequences=True),
    Dense(50, activation="softmax")
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Text generation model prepared.")

# 13. How do you implement a simple Bi-directional GRU for sequence labeling?
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=100),
    Bidirectional(GRU(64, return_sequences=True)),
    TimeDistributed(Dense(10, activation="softmax"))
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
