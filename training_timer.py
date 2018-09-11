import os
import math
import random
import csv
import nltk
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
from sklearn.svm import LinearSVC, SVC, NuSVC
from pprint import pprint
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from operator import itemgetter
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import csr_matrix
import time

start_time = time.time()
stop_words = stopwords.words("english")
def loadData(file):
  csvfile = open(file, 'r+')
  lines = csv.reader(csvfile)
  data = list(lines)
  x=np.array([point[1] for point in data if point[0]!='id'])
  y=csr_matrix([[int(point[i]) for i in range(2,len(point))] for point in data if point[0]!='id']).toarray()
  X_TRAIN, X_TEST , Y_TRAIN, Y_TEST = train_test_split(x,y,test_size=.2)
  return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST


result_array = []
mean = 0
for i in range(1):
	mean = 0
	for j in range(50):
		start_time = time.time()
		train_data, test_data, train_labels, test_labels = loadData('train.csv')
		mlb = MultiLabelBinarizer()
		binary_train_labels = mlb.fit_transform(train_labels)

		vectorizer = TfidfVectorizer()
		vectorised_train_data = vectorizer.fit_transform(train_data)

		classifier = OneVsRestClassifier(SVC())
		classifier.fit(vectorised_train_data, binary_train_labels)
		mean += (time.time()-start_time)/50
		print("iteration " + str(i) + " " + str(j) + " complete")
	result_array.append(mean)
	

for number in result_array:
	print(number)

