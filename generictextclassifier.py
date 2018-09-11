import os
import math
import random
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from pprint import pprint
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.corpus import reuters
from sklearn.model_selection import train_test_split
from operator import itemgetter
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from scipy.sparse import csr_matrix
from skmultilearn.problem_transform import LabelPowerset, BinaryRelevance
import scipy
from scipy.io import arff
import time

stop_words = stopwords.words("english")

def loadData(file):
  csvfile = open(file, 'r+')
  lines = csv.reader(csvfile)
  data = list(lines)
  x=np.array([point[1] for point in data[:600] if point[0]!='id'])
  y=np.array([[int(point[i]) for i in range(2,len(point))] for point in data[:600] if point[0]!='id'])
  X_TRAIN, X_TEST , Y_TRAIN, Y_TEST = train_test_split(x,y,test_size=.2)
  return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

start_time = time.time()
train_data, test_data, train_labels, test_labels = loadData('train.csv')


mlb = MultiLabelBinarizer()
binary_train_labels = mlb.fit_transform(train_labels)
binary_test_labels = mlb.transform(test_labels)

vectorizer = TfidfVectorizer(stop_words=stop_words, decode_error="ignore")
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

accuracy = accuracy_score(test_labels, predictions)
micro_precision = precision_score(test_labels, predictions, average='micro')
micro_recall = recall_score(test_labels, predictions, average='micro')
micro_f1 = f1_score(test_labels, predictions, average='micro')

macro_precision = precision_score(test_labels, predictions, average='macro')
macro_recall = recall_score(test_labels, predictions, average='macro')
macro_f1 = f1_score(test_labels, predictions, average='macro')

runtime = time.time()-start_time

print(str(accuracy) + "\n" + str(micro_precision) + "\n" + 
	str(micro_recall) + "\n" + str(micro_f1) + "\n" + str(macro_precision) +
	"\n" + str(macro_recall) + "\n" + str(macro_f1) + "\n" + str(runtime))