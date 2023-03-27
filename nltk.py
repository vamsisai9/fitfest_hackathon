import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download required nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define keywords to search for
keywords = ['total', 'amount', 'due']

# Tokenize extracted text
tokens = word_tokenize(text)

# Remove stopwords from tokens
stop_words = set(stopwords.words('english'))
tokens = [token.lower() for token in tokens if token.lower() not in stop_words]

# Lemmatize tokens
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(token) for token in tokens]

# Convert tokens to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens)
sequences = tokenizer.texts_to_sequences(tokens)

# Pad sequences
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Load deep learning model
model = tf.keras.models.load_model('model.h5')

# Predict keyword presence using deep learning model
predictions = model.predict(padded_sequences)

# Print keywords that are present in the extracted text
for i, prediction in enumerate(predictions):
    if prediction[0] > 0.5 and tokenizer.index_word[i+1] in keywords:
        print(tokenizer.index_word[i+1])
