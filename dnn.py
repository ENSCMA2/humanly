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

vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
vectorised_train_data = vectorizer.fit_transform(train_data)
vectorised_test_data = vectorizer.transform(test_data)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=vectorised_train_data.shape[1]))
model.add(Dropout(.5))
model.add(Dense(64))
model.add(Dropout(.5))
model.add(Dense(len(train_labels[0]), activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(vectorised_train_data, train_labels, epochs=20, batch_size=128)
score = model.evaluate(vectorised_test_data, test_labels, batch_size=128)

print(score)

nn_preds = model.predict(vectorised_test_data)

nn_preds_rounded = (nn_preds > .5).astype(int)

accuracy = accuracy_score(test_labels, nn_preds_rounded)
precision = precision_score(test_labels, nn_preds_rounded, average='micro')
recall = recall_score(test_labels, nn_preds_rounded, average='micro')
f1 = f1_score(test_labels, nn_preds_rounded, average='micro')
print("Accuracy = {}".format(accuracy))
print("Precision = {}".format(precision))
print("Recall = {}".format(recall))
print("F1 = {}".format(f1))

accuracy = accuracy_score(test_labels, nn_preds_rounded)
precision = precision_score(test_labels, nn_preds_rounded, average='macro')
recall = recall_score(test_labels, nn_preds_rounded, average='macro')
f1 = f1_score(test_labels, nn_preds_rounded, average='macro')
print("Accuracy = {}".format(accuracy))
print("Precision = {}".format(precision))
print("Recall = {}".format(recall))
print("F1 = {}".format(f1))