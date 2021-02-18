# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:44:06 2021

@author: Ahsen Yavas
"""

import pandas as pd
import numpy as np

comments = pd.read_csv('Restaurant_Reviews.csv', error_bad_lines=False)

import re
import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

#Preprocessing
collection = []
for i in range(1000):
    comment = re.sub('[^a-zA-Z]',' ',comments['Review'][i])
    comment = comment.lower()
    comment = comment.split()
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    collection.append(comment)

#Feautre Extraction
#Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)

X = cv.fit_transform(collection).toarray() 
y = comments.iloc[:,1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)






















