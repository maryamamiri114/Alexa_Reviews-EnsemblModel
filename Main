import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.model_selection import learning_curve

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag

import logging

from gensim.models import Word2Vec

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer
from keras.layers.convolutional import Convolution1D
from keras.layers import MaxPool1D, Flatten
from keras import callbacks


from keras.layers.embeddings import Embedding
import tensorflow as tf

from keras.regularizers import l1, l2, l1_l2

import math
import random
from collections import defaultdict
from pprint import pprint

import warnings

warnings.filterwarnings(action='ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud

# Set global styles for plots
sns.set_style(style='white')
sns.set_context(context='notebook', font_scale=1.3, rc={'figure.figsize': (16, 9)})

# df = pd.read_excel('~/Desktop/Project/1/tweets.xls')

df = pd.read_csv('Alexa.tsv', delimiter='\t', encoding='utf-8')

wordcloud = WordCloud(background_color='white',max_words=200,
                          max_font_size=40).generate(str(df['verified_reviews']))

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

vect = CountVectorizer(max_features=1000, binary=True)
X = vect.fit_transform(df.verified_reviews).toarray

# print(X)

X = df.verified_reviews
y = df.feedback

plt.hist(y, bins=10)
plt.show()

cv = ShuffleSplit(n_splits=20, test_size=0.2)

models = [
    LinearSVC(C=0.1, max_iter=500),
    SVC(kernel='rbf')
]

sm = SMOTE()

# Init a dictionary for storing results of each run for each model
results = {
    model.__class__.__name__: {
        'accuracy': [],
        'f1_score': [],
        'confusion_matrix': [],
        'cost': []
    } for model in models
}

for train_index, test_index in cv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_vect = vect.fit_transform(X_train)
    X_test_vect = vect.transform(X_test)

    X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)

    for model in models:
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_vect)

        acc = accuracy_score(y_test, y_pred)
        # f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[model.__class__.__name__]['accuracy'].append(acc)
        # results[model.__class__.__name__]['f1_score'].append(f1)
        results[model.__class__.__name__]['confusion_matrix'].append(cm)

for model, d in results.items():
    avg_acc = sum(d['accuracy']) / len(d['accuracy']) * 100
    # avg_f1 = sum(d['f1_score']) / len(d['f1_score']) * 100
    avg_cm = sum(d['confusion_matrix']) / len(d['confusion_matrix'])

    slashes = '-' * 30

    s = f"""{model}\n{slashes}
        Avg. Accuracy: {avg_acc:.2f}%
        Avg. F1 Score: {avg_acc:.2f}
        Avg. Confusion Matrix: 
        \n{avg_cm}
        """
    print(s)


df = pd.read_csv('Alexa.tsv', delimiter='\t', encoding='utf-8')

# print(df.head())


# Split data into training set and validation
X_train, X_test, y_train, y_test = train_test_split(df['verified_reviews'], df['feedback'], \
                                                    test_size=0.10)

print('Load %d training examples and %d validation examples. \n' % (X_train.shape[0], X_test.shape[0]))


def cleanText(raw_text, remove_stopwords=False, stemming=False, split_text=False):
    '''
    Convert a raw review to a cleaned review
    '''
    # text = BeautifulSoup(raw_text, 'lxml').get_text()  # remove html
    text = raw_text
    letters_only = re.sub("[^a-zA-Z]", " ", text)  # remove non-character
    words = letters_only.lower().split()  # convert to lower case

    if remove_stopwords:  # remove stopword
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    if stemming == True:  # stemming
        #         stemmer = PorterStemmer()
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]

    if split_text == True:  # split text
        return (words)

    return (" ".join(words))


X_train_cleaned = []
X_test_cleaned = []

for d in X_train:
    X_train_cleaned.append(cleanText(d))

for d in X_test:
    X_test_cleaned.append(cleanText(d))

# Fit and transform the training data to a document-term matrix using CountVectorizer
countVect = CountVectorizer()
X_train_countVect = countVect.fit_transform(X_train_cleaned)
print("Number of features : %d \n" % len(countVect.get_feature_names()))  # 6378
print("Show some feature names : \n", countVect.get_feature_names()[::1000])

# Train MultinomialNB classifier
mnb = MultinomialNB()
mnb.fit(X_train_countVect, y_train)


def modelEvaluation(predictions):
    '''
    Print model evaluation to predicted result
    '''
    print("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("\nAUC score : {:.4f}".format(roc_auc_score(y_test, predictions)))
    print("\nClassification report : \n", metrics.classification_report(y_test, predictions))
    print("\nConfusion Matrix : \n", metrics.confusion_matrix(y_test, predictions))


# Evaluate the model on validaton set
predictions = mnb.predict(countVect.transform(X_test_cleaned))
modelEvaluation(predictions)

# Fit and transform the training data to a document-term matrix using TfidfVectorizer
tfidf = TfidfVectorizer(min_df=5)  # minimum document frequency of 5
X_train_tfidf = tfidf.fit_transform(X_train)
print("Number of features : %d \n" % len(tfidf.get_feature_names()))  # 1722
print("Show some feature names : \n", tfidf.get_feature_names()[::1000])

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)

# Look at the top 10 features with smallest and the largest coefficients
feature_names = np.array(tfidf.get_feature_names())
sorted_coef_index = lr.coef_[0].argsort()
print('\nTop 10 features with smallest coefficients :\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Top 10 features with largest coefficients : \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

# Evaluate on the validaton set
predictions = lr.predict(tfidf.transform(X_test_cleaned))
modelEvaluation(predictions)

# Building a pipeline
estimators = [("tfidf", TfidfVectorizer()), ("lr", LogisticRegression())]
model = Pipeline(estimators)

# Grid search
params = {"lr__C": [0.1, 1, 10],  # regularization param of logistic regression
          "tfidf__min_df": [1, 3],  # min count of words
          "tfidf__max_features": [1000, None],  # max features
          "tfidf__ngram_range": [(1, 1), (1, 2)],  # 1-grams or 2-grams
          "tfidf__stop_words": [None, "english"]}  # use stopwords or don't

grid = GridSearchCV(estimator=model, param_grid=params, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_cleaned, y_train)
print("The best paramenter set is : \n", grid.best_params_)

# Evaluate on the validaton set
predictions = grid.predict(X_test_cleaned)
modelEvaluation(predictions)

# Split review text into parsed sentences uisng NLTK's punkt tokenizer
# nltk.download()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def parseSent(review, tokenizer, remove_stopwords=False):
    '''
    Parse text into sentences
    '''
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(cleanText(raw_sentence, remove_stopwords, split_text=True))
    return sentences


# Parse each review in the training set into sentences
sentences = []
for review in X_train_cleaned:
    sentences += parseSent(review, tokenizer)

print('%d parsed sentence in the training set\n' % len(sentences))
print('Show a parsed sentence in the training set : \n', sentences[10])

# Fit parsed sentences to Word2Vec model
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

num_features = 10000  # embedding dimension
min_word_count = 10
num_workers = 4
context = 10
downsampling = 1e-3

print("Training Word2Vec model ...\n")
w2v = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, \
               window=context, sample=downsampling)
w2v.init_sims(replace=True)
w2v.save("w2v_300features_10minwordcounts_10context")  # save trained word2vec model

print("Number of words in the vocabulary list : %d \n" % len(w2v.wv.index2word))  # 4016
print("Show first 10 words in the vocalbulary list  vocabulary list: \n", w2v.wv.index2word[0:10])


# Transfrom the training data into feature vectors

def makeFeatureVec(review, model, num_features):
    '''
    Transform a review to a feature vector by averaging feature vectors of words
    appeared in that review and in the volcabulary list created
    '''
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word)  # index2word is the volcabulary list of the Word2Vec model
    isZeroVec = True
    for word in review:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
            isZeroVec = False
    if isZeroVec == False:
        featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    '''
    Transform all reviews to feature vectors using makeFeatureVec()
    '''
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


# Get feature vectors for training set
X_train_cleaned = []
for review in X_train:
    X_train_cleaned.append(cleanText(review, remove_stopwords=True, split_text=True))
trainVector = getAvgFeatureVecs(X_train_cleaned, w2v, num_features)
print("Training set : %d feature vectors with %d dimensions" % trainVector.shape)

# Get feature vectors for validation set
X_test_cleaned = []
for review in X_test:
    X_test_cleaned.append(cleanText(review, remove_stopwords=True, split_text=True))
testVector = getAvgFeatureVecs(X_test_cleaned, w2v, num_features)
print("Validation set : %d feature vectors with %d dimensions" % testVector.shape)

# debugging
# print("Checkinf for NaN and Inf")
# print("np.inf=", np.where(np.isnan(trainVector)))
# print("is.inf=", np.where(np.isinf(trainVector)))
# print("np.max=", np.max(abs(trainVector)))

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(trainVector, y_train)
predictions = rf.predict(testVector)
modelEvaluation(predictions)



top_words = 2000
maxlen = 100
batch_size = 16
nb_classes = 2
nb_epoch = 15


# Vectorize X_train and X_test to 2D tensor
tokenizer = Tokenizer(nb_words=top_words)  # only consider top 20000 words in the corpse
tokenizer.fit_on_texts(X_train)
# tokenizer.word_index #access word-to-index dictionary of trained tokenizer

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

X_train_seq = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test_seq = sequence.pad_sequences(sequences_test, maxlen=maxlen)

# one-hot encoding of y_train and y_test
y_train_seq = np_utils.to_categorical(y_train, nb_classes)
y_test_seq = np_utils.to_categorical(y_test, nb_classes)

print('X_train shape:', X_train_seq.shape)  # (27799, 100)
print('X_test shape:', X_test_seq.shape)  # (3089, 100)
print('y_train shape:', y_train_seq.shape)  # (27799, 2)
print('y_test shape:', y_test_seq.shape)  # (3089, 2)



# Construct a simple LSTM
model1 = Sequential()
model1.add(Embedding(top_words, 8, input_length=maxlen))
model1.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l1_l2(0.01), recurrent_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01)))
model1.add(Dense(nb_class, eskernel_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01)))
model1.add(Activation('softmax'))
model1.summary()

# Compile LSTM
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model1.fit(X_train_seq, y_train_seq, validation_split=0.2, batch_size=batch_size, epochs=nb_epoch, verbose=1)

plt.plot(history.history['loss'], label='test set')
plt.plot(history.history['val_loss'], label='train set')
plt.title(1)
plt.ylabel('Accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()


# Model evluation
score = model1.evaluate(X_test_seq, y_test_seq, batch_size=batch_size)
print('Test loss : {:.4f}'.format(score[0]))
print('Test accuracy : {:.4f}'.format(score[1]))

top_words = 2000
maxlen = 100
batch_size = 16
nb_classes = 2
nb_epoch = 15

model22 = Sequential()
model22.add(Embedding(top_words, 8, input_length=maxlen))
model22.add(Convolution1D(64, 3, padding='valid', activation='sigmoid', kernel_regularizer=l1_l2(0.01), recurrent_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01)))
model22.add(MaxPool1D())
model22.add(Flatten())
model22.add(Dense(10))
model22.add(Dense(2))
model22.add(Activation('softmax'))


print(model22.summary())

model22.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model22.fit(X_train_seq, y_train_seq, validation_split=0.2, batch_size=batch_size, epochs=nb_epoch, verbose=1)
score = model22.evaluate(X_test_seq, y_test_seq, batch_size=batch_size)
print('Test loss : {:.4f}'.format(score[0]))
print('Test accuracy : {:.4f}'.format(score[1]))

plt.plot(history.history['val_accuracy'], label='CNN')
#plt.plot(history.history['val_loss'], label='train set')
plt.title(22)
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="lower right")
plt.show()


# Load trained Word2Vec model
w2v = Word2Vec.load("w2v_300features_10minwordcounts_10context")

# Get Word2Vec embedding matrix
embedding_matrix = w2v.wv.syn0  # embedding matrix, type = numpy.ndarray
print("Shape of embedding matrix : ", embedding_matrix.shape)  # (4016, 300) = (volcabulary size, embedding dimension)
# w2v.wv.syn0[0] #feature vector of the first word in the volcabulary list

top_words = embedding_matrix.shape[0]  # 4016
maxlen = 100
batch_size = 16
nb_classes = 2
nb_epoch = 50

# Vectorize X_train and X_test to 2D tensor
tokenizer = Tokenizer(num_words=top_words)  # only consider top 20000 words in the corpse
tokenizer.fit_on_texts(X_train)
# tokenizer.word_index #access word-to-index dictionary of trained tokenizer

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

X_train_seq = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test_seq = sequence.pad_sequences(sequences_test, maxlen=maxlen)

# one-hot encoding of y_train and y_test
y_train_seq = np_utils.to_categorical(y_train, nb_classes)
y_test_seq = np_utils.to_categorical(y_test, nb_classes)

print('X_train shape:', X_train_seq.shape)  # (27799, 100)
print('X_test shape:', X_test_seq.shape)  # (3089, 100)
print('y_train shape:', y_train_seq.shape)  # (27799, 2)
print('y_test shape:', y_test_seq.shape)  # (3089, 2)

# Construct Word2Vec embedding layer
embedding_layer = Embedding(embedding_matrix.shape[0],  # 4016
                            embedding_matrix.shape[1],  # 300
                            weights=[embedding_matrix])

# Construct LSTM with Word2Vec embedding
model2 = Sequential()
model2.add(embedding_layer)
model2.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2kernel_regularizer=l1_l2(0.01), recurrent_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01)))
model2.add(Dense(nb_classes, kernel_regularizer=l1_l2(0.01), recurrent_regularizer=l1_l2(0.01), bias_regularizer=l1_l2(0.01)))
model2.add(Activation('softmax'))
model2.summary()

# Compile model
model2.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])

history = model2.fit(X_train_seq, y_train_seq, validation_split=0.2, batch_size=batch_size, epochs=nb_epoch, verbose=1)

plt.plot(history.history['val_accuracy'], label='LSTM')
#plt.plot(history.history['val_loss'], label='MAE (validation data)')
plt.title(2)
plt.ylabel('accuracy')
plt.xlabel('No. epoch')
plt.xlim(0, 1)
plt.legend(loc="lower right")
plt.show()

# Model evaluation
score = model2.evaluate(X_test_seq, y_test_seq, batch_size=batch_size)
print('Test loss : {:.4f}'.format(score[0]))
print('Test accuracy : {:.4f}'.format(score[1]))

top_words = embedding_matrix.shape[0]  # 4016
maxlen = 100
batch_size = 16
nb_classes = 2
nb_epoch = 5

# Vectorize X_train and X_test to 2D tensor
tokenizer = Tokenizer(num_words=top_words)  # only consider top 20000 words in the corpse
tokenizer.fit_on_texts(X_train)
# tokenizer.word_index #access word-to-index dictionary of trained tokenizer

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

X_train_seq = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test_seq = sequence.pad_sequences(sequences_test, maxlen=maxlen)

y_train_seq = np_utils.to_categorical(y_train, nb_classes)
y_test_seq = np_utils.to_categorical(y_test, nb_classes)

model22 = Sequential()
model22.add(embedding_layer)
model22.add(Convolution1D(16, 3, padding='valid', activation='relu'))
model22.add(MaxPool1D())
model22.add(Convolution1D(16, 3, padding='valid', activation='relu'))
model22.add(MaxPool1D())
model22.add(Flatten())
model22.add(Dense(10))
model22.add(Dense(2))

print(model.summary())

model22.summary()

model22.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model22.fit(X_train_seq, y_train_seq, validation_split=0.2, batch_size=batch_size, epochs=nb_epoch, verbose=1)

plt.plot(history.history['loss'], label='MAE (training data)')
plt.plot(history.history['val_loss'], label='MAE (validation data)')
plt.title(2)
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")
plt.show()
