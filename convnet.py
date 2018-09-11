import csv
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop

def loadData(file):
  csvfile = open(file, 'r+')
  lines = csv.reader(csvfile)
  data = list(lines)
  x=np.array([point[1] for point in data if point[0]!='id'], dtype=object)
  y=np.array([[int(point[i]) for i in range(2,len(point))] for point in data if point[0]!='id'], dtype=object)
  print(y[0])
  X_TRAIN, X_TEST , Y_TRAIN, Y_TEST = train_test_split(x,y,test_size=.2)
  return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

train_data, test_data, train_labels, test_labels = loadData('train.csv')

max_len = max(len(X[i]) for i in range(X.shape[0]))
print ("Max length = {}".format(max_len))

wordslist = " ".join(X).split()
wordslist = list(set(wordslist))
print (wordslist[:100], len(wordslist))

word_indices = dict((c, i) for i, c in enumerate(wordslist))


maxlen = 500
X_seq = np.zeros((len(X), maxlen))
for i, msg in enumerate(X):
    for t, word in enumerate(msg):
        if t < maxlen:
            try:
                X_seq[i, t] = word_indices[word]
            except KeyError:
                X_seq[i, t] = -1
        else:
            continue

