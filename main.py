from collections import defaultdict

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

model = Word2Vec(sentences=filtered_tokens_lemmas, vector_size=100, window=5, min_count=1, sg=0)


word_vectors = []
for sentence in filtered_tokens_lemmas:
    for word in sentence:
        if word in model.wv:
            word_vectors.append(model.wv[word])

if word_vectors:
    document_vector = np.mean(word_vectors, axis=0)
else:
    document_vector = np.zeros(model.vector_size)


# Print the document vector
print("Document Vector (Word2Vec):")
print(document_vector)


##### 6) what is GloVe? Chatgpt answer #####

##  GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm for obtaining vector representations for words.
##  These representations, often referred to as word embeddings, capture semantic meanings of words based on the context in which they appear.
##  GloVe was developed by Stanford researchers Jeffrey Pennington, Richard Socher, and Christopher D. Manning.


##### 7) Tagging by CYK ########

#### Manual example in read me

#### Code examples:

def cyk_parse(sentence, grammar):
    # n = len(sentence)
    # table = [[set() for _ in range(n+1)] for _ in range(n+1)]

    # Step 1: Tokenization
    tokens = sentence.split()
    n = len(tokens)
    table = [[set() for _ in range(n + 1)] for _ in range(n + 1)]

    # Step 2: Initialization
    for i in range(1, n + 1):
        for rule in grammar:
            if rule[1] == tokens[i - 1]:
                table[i][i].add(rule[0])

    # Step 3: Rule Application
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            for k in range(i, j):
                for rule in grammar:
                    if len(rule) == 3:
                        for left in table[i][k]:
                            for right in table[k + 1][j]:
                                if rule[1] in left and rule[2] in right:
                                    table[i][j].add(rule[0])

    # Step 4: Backtracking
    if 'S' in table[1][n]:
        return True, table
    else:
        return False, table


sentence_to_CYK1 = "The shooter says goodbye to his love"

grammar1 = [
    ('S', 'NP', 'VP'),
    ('NP', 'Det', 'Noun'),
    ('NP', 'Possessive', 'Noun'),
    ('VP', 'Verb', 'NP', 'PP'),
    ('Det', 'The'),
    ('Noun', 'shooter'),
    ('Noun', 'love'),
    ('Possessive', 'his'),
    ('Verb', 'says'),
    ('Verb', 'goodbye'),
    ('PP', 'Preposition', 'NP'),
    ('Preposition', 'to')
]

parsed1, table1 = cyk_parse(sentence_to_CYK1, grammar1)

if parsed1:
    print("Input sentence: ", sentence_to_CYK1)
    print("Parse table: ")
    for row in table1:
        print(row)
else:
    print("Input sentence: ", sentence_to_CYK1)
    print("Sentence not parsed.")



sentence_to_CYK2 = "Books read quickly in libraries"

grammar2 = [
    ('S', 'NP', 'VP'),
    ('NP', 'Det', 'Noun'),
    ('VP', 'Verb', 'NP'),
    ('Det', 'Books'),
    ('Det', 'a'),
    ('Noun', 'read'),
    ('Noun', 'libraries'),
    ('Verb', 'read'),
    ('Adv', 'quickly'),
    ('Prep', 'in'),
]

parsed2, table2 = cyk_parse(sentence_to_CYK2, grammar2)

if parsed2:
    print("Input sentence: ", sentence_to_CYK2)
    print("Parse table: ")
    for row in table2:
        print(row)
else:
    print("Input sentence: ", sentence_to_CYK2)
    print("Sentence not parsed.")

sentence_to_CYK3 = "She sings beautifully"

grammar3 = [
    ('S', 'NP', 'VP'),
    ('NP', 'Pronoun'),
    ('VP', 'Verb', 'Adv'),
    ('Pronoun', 'She'),
    ('Verb', 'sings'),
    ('Adv', 'beautifully'),
]

parsed3, table3 = cyk_parse(sentence_to_CYK3, grammar3)

if parsed3:
    print("Input sentence: ", sentence_to_CYK3)
    print("Parse table: ")
    for row in table3:
        print(row)
else:
    print("Input sentence: ", sentence_to_CYK3)
    print("Sentence not parsed.")