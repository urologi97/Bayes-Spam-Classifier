from __future__ import division
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


import pandas as pd
import numpy as np
import re


df = pd.read_table('sms', names=['class', 'text'])
df = df[['text', 'class']]

df.head()

f = open("stopwords.txt", "r")
stopwords = f.read().split("\n")

def stopword_removal(df, stopwords):
    row, col = df.shape
    for i in range(row):
        sentence = df.loc[i]['text']
        cleaned = []
        for word in sentence.split():
            if word not in stopwords:
                cleaned.append(word)
        df.loc[i]['text'] = ' '.join(cleaned)
    return df


def tokenize(df):
    row, col = df.shape
    for i in range(row):
        sentence = df.loc[i]['text']
        lowercase = sentence.lower() # case folding
        regex = re.compile(r'[^a-zA-Z]')
        alphanumeric = re.sub(regex, ' ', lowercase) # remove if not alphabet
        stripped = ' '.join(alphanumeric.split()) # strip whitespace
        df.loc[i]['text'] = stripped
    return df

def stem(df):
    stemmer = SnowballStemmer('english')
    row, col = df.shape
    for i in range(row):
        sentence = df.loc[i]['text']
        stemed_words = []
        for word in sentence.split():
            stemed_word = stemmer.stem(word)
            stemed_words.append(stemed_word)
        df.loc[i]['text'] = ' '.join(stemed_words)
    return df


def get_unique_words(df):
    row, col = df.shape
    unique_words = set()
    for i in range(row):
        sentence = df.loc[i]['text']
        for word in sentence.split():
            unique_words.add(word)
    return unique_words

def get_tf_matrix(df, unique_words):
    tf_matrix = []
    row, col = df.shape
    word_freq = {}

    #set frequency to 0 for all term
    for word in unique_words:
        word_freq[word] = 0

    for i in range(row):
        sentence = df.loc[i]['text']
        freq = word_freq.copy()

        # increment frequency if word found in sentence
        for word in sentence.split():
            freq[word] += 1

        tf_matrix.append(freq)
    return tf_matrix

def tf_matrix_to_array(tf_matrix):
    tf_matrix_array = []
    row = len(tf_matrix)
    for i in range(row):
        feature = np.array(list(tf_matrix[i].values()))
        tf_matrix_array.append(feature)
    return np.vstack(tf_matrix_array)

# preprocess
tokenized = tokenize(df)
filtered = stopword_removal(tokenized, stopwords)
stemed = stem(filtered)
unique_words = get_unique_words(df)
tf_matrix = get_tf_matrix(df, unique_words)
X = tf_matrix_to_array(tf_matrix)

#split training and training data set
X_train, X_test, y_train, y_test = train_test_split(X, df['class'], test_size=0.75)

# create and train mnb classifier
classifier = MultinomialNB() # buat classifier
classifier.fit(X_train, y_train)  # train the classifier
prediction = classifier.predict(X_test)


# measure accuracy
acc = pd.DataFrame({
        'prediction': prediction,
        'label': y_test
        })

pos_case = acc[acc['label'] == 'spam']
neg_case = acc[acc['label'] == 'ham']
tp = np.sum(pos_case['prediction'] == 'spam')
tn = np.sum(neg_case['prediction'] == 'ham')
fp = np.sum(neg_case['prediction'] == 'spam')
fn = np.sum(pos_case['prediction'] == 'ham')
print(tp)
print(tn)
print(fp)
print(fn)
precision =  tp/(tp+fp)
print (precision)
recall = tp/(tp+fn)
print (recall)
f1_score = 2 * ((precision * recall) / (precision + recall))
print("Hasil precision: %.2f%%\nHasil Recall: %.2f%%\nF1 Score: %.2f%%" % (precision*100, recall*100, f1_score * 100))

# example
sample = pd.DataFrame({'text': [
       "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
]})

tokenized = tokenize(sample)
filtered = stopword_removal(tokenized, stopwords)
stemed = stem(filtered)
tf_matrix = get_tf_matrix(stemed, unique_words)
tf_array = tf_matrix_to_array(tf_matrix)
result = classifier.predict(tf_array) # ubah input term frequency ke array dan lakukan prediksi
print(result)

