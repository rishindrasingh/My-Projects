# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 08:54:12 2019

@author: Rishindra
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')



wordnet=WordNetLemmatizer()

corpus1=[]
corpus2=[]

#lematizing for training dataset
for i in range(0,len(train)):
    review=re.sub('[^a-zA-Z]', ' ',train['tweet'][i])
    review=review.lower()
    review=review.split()
    review=[wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus1.append(review)



#lematizing for testing dataset
for i in range(0,len(test)):
    review=re.sub('[^a-zA-Z]', ' ',test['tweet'][i])
    review=review.lower()
    review=review.split()
    review=[wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus2.append(review)



#tf-idf model for training set
    
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer(max_features=7875)
X_train=tf.fit_transform(corpus1).toarray()

#tf-idf model for testing set
    
from sklearn.feature_extraction.text import TfidfVectorizer
tf1=TfidfVectorizer(max_features=7875)
X_test=tf1.fit_transform(corpus2).toarray()

y_train=train.iloc[:,1].values


#creating model
import xgboost

from xgboost import XGBClassifier
classifier=XGBClassifier(n_estimators=10)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)


pred=pd.DataFrame(y_pred)

sub_df=pd.read_csv('sample_submission.csv')



dataset=pd.concat([sub_df['id'],pred],axis=1)
dataset.columns=['id','label']
dataset.to_csv('sample_submission01.csv',index=False)
