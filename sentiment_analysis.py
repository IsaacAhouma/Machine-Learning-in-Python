# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:25:41 2017

@author: Isaac
"""

from __future__ import division
import sklearn.cross_validation
import sklearn.linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import scipy


def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)

def sentiment_analysis(data):
    products = pd.read_csv('amazon_baby.csv')
    products.head()
    products = products.fillna({'review':''})  # fill in N/A's in the review column
    
    products['review_clean'] = products['review'].apply(remove_punctuation)
    #products = products[products['rating'] != 3]
    #products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
    #products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1 if rating < 3 else 0)
    products['sentiment'] = products['rating'].apply(lambda rating : 'Positive' if rating > 3 else 'Negative' if rating < 3 else 'Neutral')
    
    products = products.sample(len(products))
    
    products.head()
    
    train_data,test_data=sklearn.cross_validation.train_test_split(products,train_size=0.8,random_state=1)
    
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
         # Use this token pattern to keep single-letter words
    # First, learn vocabulary from the training data and assign columns to words
    # Then convert the training data into a sparse matrix
    train_matrix = vectorizer.fit_transform(train_data['review_clean'])
    # Second, convert the test data into a sparse matrix, using the same word-column mapping
    test_matrix = vectorizer.transform(test_data['review_clean'])
    
    
    sentiment_model = LogisticRegression()
    
    sentiment_model.fit(train_matrix, train_data['sentiment'])
    
    sentiment_model.coef_
    
    
#    print np.sum(sum(sentiment_model.coef_ >= 0))
    
    sample_test_data = test_data[0:100]
#    print sample_test_data
    
    sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
    
    sample_test_prediction = pandas.Series(sentiment_model.predict(sample_test_matrix))
#    print scores
    df = pandas.Series(data)
    df.apply(remove_punctuation)
    if type(data)!=scipy.sparse.csr.csr_matrix:
        processed_data = vectorizer.transform(df)
    
    test_accuracy = sum(sample_test_data['sentiment'] == sample_test_prediction ) / len(sample_test_data)
    print test_accuracy
    return pandas.Series(sentiment_model.predict(processed_data))