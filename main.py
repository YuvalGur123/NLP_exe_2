import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, RegexpTokenizer
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models.word2vec import Word2Vec

### First text processing as done in exercise 1 ###
f = open("example_text.txt")
string_text = f.read()

sentence_tokens = nltk.sent_tokenize(string_text)
word_tokenizer = RegexpTokenizer(r'\b\w+\b')

tokens_list = [nltk.word_tokenize(sentence) for sentence in sentence_tokens]
tokens_flat = [token for sentence_tokens in tokens_list for token in sentence_tokens]

tokens = word_tokenizer.tokenize(' '.join(tokens_flat))


tokens_lower = [x.lower() for x in tokens]
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else lemmatizer.lemmatize(i) for i, j in pos_tag(tokens_lower)]

ps = PorterStemmer()
stemmed_text = [ps.stem(token) for token in tokens_lower]

########## 4) removing stop word #############

stop_words = set(stopwords.words('english'))

filtered_tokens_lemmas = [token for token in lemmas if token not in stop_words]

filtered_tokens_stem = [token for token in stemmed_text if token not in stop_words]

######## 5) Feature extraction ########

### 1) BoW

text = " ".join(filtered_tokens_lemmas)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform([text])

features_BoW = vectorizer.get_feature_names_out()

bow_dict = {}
for token, idx in vectorizer.vocabulary_.items():
    bow_dict[token] = X[0, idx]


print("Bag of Words representation:")
print(bow_dict)

### 2) TF-IDF

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([text])

tokens = vectorizer.get_feature_names_out()

tfidf_dict = {}
for token, idx in vectorizer.vocabulary_.items():
    tfidf_dict[token] = X[0, idx]

print("TF-IDF representation:")
print(tfidf_dict)

### 3) WORD2VEC

model = Word2Vec(sentences=[text], vector_size=100, window=5, min_count=1, sg=0)


def get_word_vector(word):
    try:
        return model.wv[word]
    except KeyError:
        return np.zeros(model.vector_size)


document_vector = np.mean([get_word_vector(word) for word in [text]], axis=0)

# Print the document vector
print("Document Vector (Word2Vec):")
print(document_vector)

