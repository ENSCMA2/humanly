import os
import math
import random
import csv
import tracemalloc
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.corpus import reuters
from sklearn.model_selection import train_test_split
from operator import itemgetter
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance


stop_words = stopwords.words("english")

def loadData(file):
  csvfile = open(file, 'r+')
  data = pd.read_csv(csvfile)
  x = data.comment_text
  y = data.iloc[:,2:]
  print(repr(data))

  x=np.array([point[0] for point in data[:300] if point[0]!='comment_text'])
  y=np.array([[point[i] for i in range(2,len(point))] for point in data[:300] if point[0]!='comment_text'])
  print(len(x))
  print(len(y))
  X_TRAIN, X_TEST , Y_TRAIN, Y_TEST = train_test_split(x,y,test_size=.2)
  return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST


train_data, test_data, train_labels, test_labels = loadData('training.csv')


mlb = MultiLabelBinarizer()
binary_train_labels = mlb.fit_transform(train_labels)
binary_test_labels = mlb.transform(test_labels)

vectorizer = TfidfVectorizer()
vectorised_train_data = vectorizer.fit_transform(train_data)
vectorised_test_data = vectorizer.transform(test_data)

print(vectorised_test_data)
print(vectorised_train_data)
print(train_labels)
print(test_labels)
print("ok1")
classifier = BinaryRelevance(GaussianNB())
print("ok2")

classifier.fit(vectorised_train_data.todense(), train_labels)
print("ok3")
predictions = classifier.predict(vectorised_test_data.todense())
predictions_final = predictions.todense()
print("ok4")

accuracy = accuracy_score(test_labels, predictions_final)
precision = precision_score(test_labels, predictions_final, average='micro')
recall = recall_score(test_labels, predictions_final, average='micro')
f1 = f1_score(test_labels, predictions_final, average='micro')
print("Micro-average quality numbers")
print("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(accuracy, precision, recall, f1))

accuracy = accuracy_score(test_labels, predictions_final)
precision = precision_score(test_labels, predictions_final, average='macro')
recall = recall_score(test_labels, predictions_final, average='macro')
f1 = f1_score(test_labels, predictions_final, average='macro')
print("Macro-average quality numbers")
print("Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(accuracy, precision, recall, f1))