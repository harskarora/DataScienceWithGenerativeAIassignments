"""
 1. What is the primary goal of Natural Language Processing (NLP)?
Answer: The primary goal of NLP is to enable machines to understand, interpret, and respond to human language.
Example:
goal = "Enable machines to process and analyze natural language data."
print(f"NLP Goal: {goal}")

 2. What does "tokenization" refer to in text processing?
Answer: Tokenization is the process of splitting text into smaller units like words or sentences.
Example:
text = "Hello, world! Welcome to NLP."
tokens = text.split()   Basic word tokenization
print(f"Tokens: {tokens}")

 3. What is the difference between lemmatization and stemming?
Answer:
 Stemming: Trims words to their root form without context (e.g., 'running' -> 'run').
 Lemmatization: Reduces words to their base form considering grammar (e.g., 'better' -> 'good').
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

word = "running"
print(f"Stemmed: {stemmer.stem(word)}")
print(f"Lemmatized: {lemmatizer.lemmatize(word, pos='v')}")

 4. What is the role of regular expressions (regex) in text processing?
Answer: Regex helps in pattern matching for tasks like text cleaning and validation.
Example:
import re
emails = "Contact us at support@example.com or sales@example.org."
extracted_emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', emails)
print(f"Extracted Emails: {extracted_emails}")

 5. What is Word2Vec and how does it represent words in a vector space?
Answer: Word2Vec maps words to dense vector representations based on their context in a corpus.
Example: (Using Gensim for Word2Vec)
from gensim.models import Word2Vec

sentences = [["hello", "world"], ["goodbye", "world"]]
model = Word2Vec(sentences, vector_size=10, min_count=1)
print(f"Vector for 'hello': {model.wv['hello']}")

 6. How does frequency distribution help in text analysis?
Answer: Counts occurrences of words to identify common terms.
Example:
from collections import Counter

text = "apple banana apple orange banana apple"
word_counts = Counter(text.split())
print(f"Word Counts: {word_counts}")

 7. Why is text normalization important in NLP?
Answer: It ensures consistency and reduces variations of words for better analysis.
Example:
text = "The U.S. and USA are the same."
normalized_text = text.lower().replace("u.s.", "usa")
print(f"Normalized Text: {normalized_text}")

 8. What is the difference between sentence tokenization and word tokenization?
Answer:
 Sentence Tokenization: Splits text into sentences.
 Word Tokenization: Splits sentences into words.
from nltk.tokenize import sent_tokenize, word_tokenize

paragraph = "Hello world! NLP is amazing. Let's learn."
sentences = sent_tokenize(paragraph)
words = word_tokenize(paragraph)
print(f"Sentences: {sentences}")
print(f"Words: {words}")

 9. What are co-occurrence vectors in NLP?
Answer: They represent relationships between words based on how often they appear together in a context.
Example:
from sklearn.feature_extraction.text import CountVectorizer

corpus = ["the cat sat on the mat", "the cat is black"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()
print(f"Co-occurrence Matrix:\n{X}")

 10. What is the significance of lemmatization in improving NLP tasks?
Answer: Reduces words to their meaningful base forms, improving search accuracy and reducing dimensionality.
Example:
print(f"Lemmatized (example): {lemmatizer.lemmatize('better', pos='a')}")

 11. What is the primary use of word embeddings in NLP?
Answer: Captures semantic meanings of words for better understanding in downstream tasks like classification or translation.

 12. What is an annotator in NLP?
Answer: A tool or function that labels text with metadata, such as named entities or parts of speech.
Example:
 SpaCy's annotator can label entities in text.

import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup.")
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

 13. What are the key steps in text processing before applying machine learning models?
Answer: Tokenization → Normalization → Stopword Removal → Stemming/Lemmatization → Vectorization.

 14. Why is sentence processing important in NLP?
Answer: Helps preserve sentence structure, essential for tasks like translation and summarization.

 15. How do word embeddings improve the understanding of language semantics in NLP?
Answer: They map words to vectors, capturing semantic relationships and improving contextual understanding.

 16. How does the frequency distribution of words help in text classification?
Answer: Identifies key terms that may differentiate classes in a dataset.

 17. What are the advantages of using regex in text cleaning?
Answer: Simplifies pattern matching and removal of unwanted elements.

 18. What is the difference between Word2Vec and Doc2Vec?
Answer:
 Word2Vec: Represents individual words.
 Doc2Vec: Represents entire documents in vector space.

 19. Why is understanding text normalization important in NLP?
Answer: Ensures uniformity in data for more effective processing and analysis.

 20. How does word count help in text analysis?
Answer: Provides insights into term importance for tasks like keyword extraction.

 21. How does lemmatization help in NLP tasks like search engines and chatbots?
Answer: Reduces words to base forms, ensuring accurate matching and fewer variations.

 22. What is the purpose of using Doc2Vec in text processing?
Answer: Generates vector representations for entire documents, useful for classification.

 23. What is the importance of sentence processing in NLP?
Answer: Maintains context at the sentence level, crucial for accurate analysis.

 24. What is text normalization, and what are the common techniques used in it?
Answer: Converts text to a consistent format. Techniques: lowercasing, stemming, lemmatization, removing stopwords.

 25. Why is word tokenization important in NLP?
Answer: Breaks text into meaningful units for further processing.

 26. How does sentence tokenization differ from word tokenization in NLP?
Answer: Sentence tokenization breaks text into sentences; word tokenization splits sentences into words.

 27. What is the primary purpose of text processing in NLP?
Answer: Converts raw text into structured data for analysis.

 28. What are the key challenges in NLP?
Answer: Ambiguity, language diversity, and context understanding.

 29. How do co-occurrence vectors represent relationships between words?
Answer: Encodes the frequency of words appearing together, capturing contextual relationships.

 30. What is the role of frequency distribution in text analysis?
Answer: Identifies frequently occurring terms and aids in feature selection.

 31. What is the impact of word embeddings on NLP tasks?
Answer: Enhances contextual understanding and accuracy in tasks like sentiment analysis and translation.

 32. What is the purpose of using lemmatization in text preprocessing?
Answer: Normalizes words, reducing dimensionality and improving model performance.

"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from collections import Counter
import matplotlib.pyplot as plt
import re
import spacy
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")

# 1. How can you perform word tokenization using NLTK?
sentence = "Natural Language Processing is fun and exciting!"
tokens = word_tokenize(sentence)
print(f"Word Tokens: {tokens}")

# 2. How can you perform sentence tokenization using NLTK?
paragraph = "Hello world! NLP is amazing. Let's explore it."
sentences = sent_tokenize(paragraph)
print(f"Sentence Tokens: {sentences}")

# 3. How can you remove stopwords from a sentence?
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(f"Filtered Tokens (Stopwords Removed): {filtered_tokens}")

# 4. How can you perform stemming on a word?
stemmer = PorterStemmer()
stemmed_word = stemmer.stem("running")
print(f"Stemmed Word: {stemmed_word}")

# 5. How can you perform lemmatization on a word?
lemmatizer = WordNetLemmatizer()
lemmatized_word = lemmatizer.lemmatize("running", pos="v")
print(f"Lemmatized Word: {lemmatized_word}")

# 6. How can you normalize a text by converting it to lowercase and removing punctuation?
text = "Hello, World! NLP is Awesome."
normalized_text = text.lower().translate(str.maketrans('', '', string.punctuation))
print(f"Normalized Text: {normalized_text}")

# 7. How can you create a co-occurrence matrix for words in a corpus?
corpus = ["the cat sat on the mat", "the dog lay on the rug"]
vectorizer = CountVectorizer()
co_occurrence_matrix = vectorizer.fit_transform(corpus).toarray()
print(f"Co-occurrence Matrix:\n{co_occurrence_matrix}")

# 8. How can you apply a regular expression to extract all email addresses from a text?
text_with_emails = "Contact us at support@example.com or sales@company.org."
emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text_with_emails)
print(f"Extracted Emails: {emails}")

# 9. How can you perform word embedding using Word2Vec?
sentences = [["hello", "world"], ["machine", "learning", "is", "fun"]]
word2vec_model = Word2Vec(sentences, vector_size=10, min_count=1)
print(f"Word2Vec Embedding for 'hello': {word2vec_model.wv['hello']}")

# 10. How can you use Doc2Vec to embed documents?
documents = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(["Doc one", "Doc two"])]
doc2vec_model = Doc2Vec(documents, vector_size=10, min_count=1)
print(f"Doc2Vec Embedding for Document 0: {doc2vec_model.dv[0]}")

# 11. How can you perform part-of-speech tagging?
doc = nlp("The quick brown fox jumps over the lazy dog.")
pos_tags = [(token.text, token.pos_) for token in doc]
print(f"POS Tags: {pos_tags}")

# 12. How can you find the similarity between two sentences using cosine similarity?
sentence1 = "I love programming."
sentence2 = "I enjoy coding."
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"Cosine Similarity: {similarity[0][0]}")

# 13. How can you extract named entities from a sentence?
doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")
entities = [(ent.text, ent.label_) for ent in doc.ents]
print(f"Named Entities: {entities}")

# 14. How can you split a large document into smaller chunks of text?
large_text = "This is a large document. It has multiple sentences. Let's split it."
chunks = sent_tokenize(large_text)
print(f"Chunks: {chunks}")

# 15. How can you calculate the TF-IDF (Term Frequency - Inverse Document Frequency) for a set of documents?
documents = ["NLP is great.", "NLP stands for Natural Language Processing."]
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)
print(f"TF-IDF Matrix:\n{tfidf_matrix.toarray()}")

# 16. How can you apply tokenization, stopword removal, and stemming in one go?
sentence = "This is an example sentence for NLP processing."
tokens = word_tokenize(sentence)
filtered_tokens = [stemmer.stem(word) for word in tokens if word.lower() not in stop_words]
print(f"Processed Tokens: {filtered_tokens}")

# 17. How can you visualize the frequency distribution of words in a sentence?
sentence = "Natural Language Processing is fun and exciting."
words = word_tokenize(sentence.lower())
freq_dist = Counter(words)
plt.bar(freq_dist.keys(), freq_dist.values())
plt.title("Word Frequency Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()
