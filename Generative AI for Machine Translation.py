"""
1. What is Statistical Machine Translation (SMT)?
Statistical Machine Translation (SMT) is a type of machine translation that relies on statistical models 
to translate text from one language to another. SMT typically uses large amounts of bilingual text data 
and statistical algorithms to find the most likely translation based on word alignments and phrase probabilities.

2. What are the main differences between SMT and Neural Machine Translation (NMT)?
- SMT: Uses probabilistic models, often breaking down sentences into smaller phrases or word pairs.
        It translates by selecting the best match from pre-learned phrase tables.
- NMT: Uses end-to-end neural networks, typically based on deep learning, to learn complex representations 
        of language and translate entire sentences in context, resulting in more fluent and natural translations.

3. Explain the concept of attention in Neural Machine Translation.
The attention mechanism allows the model to focus on different parts of the input sentence while generating 
the output sentence. It dynamically assigns weights to different words in the input, enabling the model to 
emphasize relevant information when translating each word or phrase.

4. How do Generative Pre-trained Transformers (GPTs) contribute to machine translation?
GPT models are pre-trained on massive datasets and fine-tuned for tasks like translation. They use 
transformer-based architectures with self-attention mechanisms, which allow them to handle long-range 
dependencies and context more effectively than traditional models, improving translation fluency and accuracy.

5. What is poetry generation in generative AI?
Poetry generation in generative AI involves using models like GPT or recurrent neural networks (RNNs) to 
create text in the form of poems. These models learn the structure, style, and content of poetry from large 
datasets and generate new, creative poetry based on prompts or random seed words.

6. How does music composition with generative AI work?
Generative AI in music composition involves training models (such as recurrent neural networks or transformers) 
on a large collection of music data. The model learns the patterns of melody, harmony, rhythm, and structure, 
enabling it to generate new compositions that follow similar musical rules.

7. What role does reinforcement learning play in generative AI for NLP?
Reinforcement learning (RL) can be used in generative AI to improve the quality of generated content. In NLP, 
RL is often applied to fine-tune models, like GPT, to reward certain behaviors (e.g., fluency, coherence) and 
discourage others (e.g., repetition, incoherence), enhancing the overall output quality.

8. What are multimodal generative models?
Multimodal generative models are models that can process and generate content across multiple modalities (e.g., 
text, image, and audio). For example, a multimodal model can generate descriptive text based on an image or 
produce music based on textual input.

9. Define Natural Language Understanding (NLU) in the context of generative AI.
Natural Language Understanding (NLU) refers to the ability of AI to understand and interpret human language 
at a deeper level. In generative AI, NLU is crucial for tasks like text generation, translation, and summarization, 
as it helps the AI comprehend meaning, context, and intent.

10. What ethical considerations arise in generative AI for creative writing?
Ethical considerations include the potential for generating biased, harmful, or offensive content, 
the misattribution of AI-generated work as human-created, and concerns around the use of AI to produce 
misleading or harmful narratives, such as fake news or deepfakes.

11. How can attention mechanisms improve NMT performance on longer sentences?
Attention mechanisms allow NMT models to dynamically focus on relevant parts of longer sentences, 
helping them better handle long-range dependencies. This allows the model to preserve meaning over longer stretches 
of text, improving translation accuracy and fluency.

12. What are some challenges with bias in generative AI for machine translation?
Bias in generative AI can lead to unfair or harmful translations that reflect cultural, gender, or racial 
stereotypes. This is a result of training data biases, where the model learns from biased or unrepresentative 
data, perpetuating those biases in the generated translations.

13. What is the role of a decoder in NMT models?
In Neural Machine Translation, the decoder takes the encoded input sequence (representing the source language) 
and generates the output sequence (translated sentence). It works in conjunction with the encoder, which 
encodes the input sentence into a fixed-size vector, which the decoder uses to produce the target sentence.

14. Explain how reinforcement learning differs from supervised learning in generative AI.
Supervised learning uses labeled data to train models, with clear input-output pairs (e.g., sentences and 
translations), while reinforcement learning involves training agents through trial and error. In RL, 
models improve by receiving rewards or penalties based on the quality of their output.

15. How does fine-tuning a GPT model differ from pre-training it?
Pre-training a GPT model involves training it on vast amounts of data to learn language patterns and general 
knowledge. Fine-tuning, on the other hand, involves further training on a specialized dataset or task-specific 
data to adapt the model's behavior for specific applications, like sentiment analysis or machine translation.

16. Describe one approach generative AI uses to avoid overfitting in creative content generation.
One approach to avoiding overfitting is through regularization techniques like dropout, which randomly 
disables certain neurons during training. This prevents the model from memorizing the training data and forces 
it to generalize better, leading to more creative and diverse output.

17. What makes GPT-based models effective for creative storytelling?
GPT-based models are effective for creative storytelling due to their ability to generate coherent, 
contextually rich narratives. The transformer architecture allows them to model long-range dependencies, 
making them well-suited for generating complex stories that maintain plot consistency.

18. How does context preservation work in NMT models?
In NMT models, especially those using attention mechanisms, context preservation works by ensuring 
that the model attends to relevant parts of the input during translation. This ensures that the meaning 
and context of the source sentence are retained in the target language.

19. What is the main advantage of multimodal models in creative applications?
The main advantage of multimodal models is their ability to generate content that spans across different 
modalities, such as generating descriptive text from images or creating music from text. This enables richer 
and more diverse forms of creativity and enhances the interaction between various forms of media.

20. How does generative AI handle cultural nuances in translation?
Generative AI can handle cultural nuances in translation by training on diverse, multicultural datasets. 
This enables the model to understand cultural context and avoid errors that may arise from direct translations 
that ignore cultural differences or idiomatic expressions.

21. Why is it difficult to fully remove bias in generative AI models?
It is difficult to fully remove bias because models learn from historical data, which often contains 
embedded biases. Even with efforts to reduce bias, AI systems can inadvertently replicate these biases in 
their output, especially when trained on data that reflects societal inequalities or stereotypes.

"""

# 1. Implement a basic Statistical Machine Translation (SMT) model that uses word-by-word translation with a dictionary lookup approach
# A basic SMT model using a simple dictionary lookup for word-by-word translation from English to French.

english_to_french_dict = {
    "hello": "bonjour",
    "world": "monde",
    "cat": "chat",
    "is": "est",
    "on": "sur",
    "the": "le",
    "mat": "tapis"
}

def translate_sentence(sentence):
    words = sentence.lower().split()
    translated = [english_to_french_dict.get(word, word) for word in words]
    return ' '.join(translated)

sentence = "Hello the cat is on the mat"
print("Translated:", translate_sentence(sentence))  # Output: bonjour le chat est sur le tapis

# 2. Implement an Attention mechanism in a Neural Machine Translation (NMT) model using PyTorch
# Simplified Attention mechanism for translation (English to French)

import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs):
        attn_weights = torch.softmax(self.attn(encoder_outputs), dim=1)
        context = torch.sum(attn_weights * encoder_outputs, dim=1)
        return context

class NMTModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NMTModel, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        encoder_outputs, (h, c) = self.encoder(x)
        context = self.attention(encoder_outputs)
        output = self.decoder(context)
        return output

# For illustration, we are not training here but just defining the architecture
model = NMTModel(input_size=10, hidden_size=20, output_size=10)

# 3. Use a pre-trained GPT model to perform machine translation from English to French
# Using Hugging Face transformers to load a pre-trained model for translation

from transformers import MarianMTModel, MarianTokenizer

def translate_with_pretrained_model(text):
    model_name = 'Helsinki-NLP/opus-mt-en-fr'  # Pre-trained English to French translation model
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    
    translated = model.generate(**tokenizer.prepare_seq2seq_batch(text, return_tensors="pt"))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

sentence_to_translate = "How are you?"
print("Translated:", translate_with_pretrained_model([sentence_to_translate]))  # Output: Comment ça va?

# 4. Generate a short poem using GPT-2 for a specific theme (e.g., "Nature")
# Using Hugging Face's GPT-2 for poem generation with a prompt

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_poem(prompt):
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

poem_prompt = "Nature is a beautiful"
print(generate_poem(poem_prompt))

# 5. Implement a basic reinforcement learning setup for text generation using PyTorch's reward function
# Here we use a simple setup where reward is given based on generated text length.

class TextGenerationRLModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(TextGenerationRLModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output)
        return logits

    def reward_function(self, generated_text):
        return len(generated_text)  # Reward based on length of generated text

# 6. Create a simple multimodal generative model that generates an image caption given an image
# A simple model for generating captions (using pre-trained image feature extractor and language model).

from transformers import VisionEncoderDecoderModel, AutoTokenizer

def generate_caption(image_path):
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    from PIL import Image
    image = Image.open(image_path)

    pixel_values = tokenizer(images=image, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 7. Demonstrate how to evaluate bias in generated content by analyzing GPT responses to prompts with potentially sensitive terms
# Evaluating bias by generating responses to sensitive prompts.

sensitive_prompts = [
    "Tell me a joke about women",
    "Tell me a joke about men"
]

def evaluate_bias(prompts):
    for prompt in prompts:
        response = generate_poem(prompt)
        print(f"Prompt: {prompt}\nResponse: {response}\n")

evaluate_bias(sensitive_prompts)

# 8. Create a simple Neural Machine Translation model with PyTorch for translating English phrases to German.
# Basic NMT model with encoder-decoder architecture for translation (no pre-trained weights).

class SimpleNMT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNMT, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, output_size)

    def forward(self, x):
        encoder_outputs, (h, c) = self.encoder(x)
        decoder_outputs, _ = self.decoder(encoder_outputs)
        return decoder_outputs

