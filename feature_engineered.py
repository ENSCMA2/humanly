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
from profanity import profanity

def loadData(file):
  csvfile = open(file, 'r+')
  lines = csv.reader(csvfile)
  data = list(lines)
  x=np.array([point[1] for point in data if point[0]!='id'])
  y=np.array([[int(point[i]) for i in range(2,len(point))] for point in data if point[0]!='id'])
  X_TRAIN, X_TEST , Y_TRAIN, Y_TEST = train_test_split(x,y,test_size=.2)
  return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

def vectorise(data):
	vectorised_data = []
	for point in data:
		total_length = len(point)
		capitals = sum(1 for c in point if c.isupper())
		caps_vs_length = float(capitals)/float(total_length)
		num_exclamation_marks = point.count('!')
		num_punctuation = sum(point.count(w) for w in '.,;:')
		num_symbols = sum(point.count(w) for w in '*&$%')
		num_words = len(point.split())
		num_unique_words = len(set(point.split()))
		words_vs_unique = num_unique_words/num_words
		num_smilies = sum(point.count(w) for w in (':-)',':)',';-)',';)'))
		contains_profanity = 0
		if profanity.contains_profanity(point):
			contains_profanity = 1
		vectorised_data.append([caps_vs_length, num_punctuation, num_symbols, words_vs_unique])
	return vectorised_data

start_time = time.time()
train_data, test_data, train_labels, test_labels = loadData('train.csv')
vectorised_train_data = vectorise(train_data)
vectorised_test_data = vectorise(test_data)

classifier = OneVsRestClassifier(LinearSVC())
classifier.fit(vectorised_train_data, train_labels)
training_time = time.time()-start_time
predictions = classifier.predict(vectorised_test_data)

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


