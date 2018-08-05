import os
import math
import random
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC, NuSVC
from pprint import pprint
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import time
from scipy.sparse import csr_matrix


stop_words = stopwords.words("english")
def elapse():
	start_time = time.time()
	while(time.time()-start_time < 1):
		continue
def loadData(file):
  csvfile = open(file, 'r+')
  lines = csv.reader(csvfile)
  data = list(lines)
  x=np.array([point[1] for point in data if point[0]!='id'])
  y=csr_matrix([[int(point[i]) for i in range(2,len(point))] for point in data if point[0]!='id']).toarray()
  X_TRAIN, X_TEST , Y_TRAIN, Y_TEST = train_test_split(x,y,test_size=.2)
  return X_TRAIN, X_TEST, Y_TRAIN, Y_TEST

def answer(predictions):
	if predictions == [0,0,0,0,0,0]:
		print("Cool!")
		return
	if predictions[0] == 1:
		print("Sounds toxic to me.")
		elapse()
	if predictions[1] == 1:
		print("In fact, that was really severely toxic. I'm hurt.")
		elapse()
	if predictions[2] == 1:
		print("Whoa! That was obscene! Maybe back off...")
		elapse()
	if predictions[3] == 1:
		print("Was that a threat? Because I think it was a threat. I'm scared now.")
		elapse()
	if predictions[4] == 1:
		print("Wow, that was quite a stinging insult. Ouch.")
		elapse()
	if predictions[5] == 1:
		print("It's who I am! Stop hating on my identity!")
		elapse()
	print("If you want to stop chatting, just type in 'quit' and I'll shut up.")
	print("\n")

train_data, test_data, train_labels, test_labels = loadData('train.csv')
mlb = MultiLabelBinarizer()
binary_train_labels = mlb.fit_transform(train_labels)
binary_test_labels = mlb.transform(test_labels)

vectorizer = TfidfVectorizer(lowercase=False,sublinear_tf=True)
vectorised_train_data = vectorizer.fit_transform(train_data)

classifier = OneVsRestClassifier(LinearSVC())
classifier.fit(vectorised_train_data, train_labels)
print("I'm trained! Now let's start talking!")
user_input = input("Tell me something!\n\n")
user_data = np.array([user_input])
vectorised_user_data = vectorizer.transform(user_data)

while(user_input != "quit"):
	predictions = classifier.predict(vectorised_user_data)
	answer(predictions[0].tolist())
	user_input = input("Tell me something else!\n")
	user_data = np.array([user_input])
	vectorised_user_data = vectorizer.transform(user_data)
print("Bye!")




