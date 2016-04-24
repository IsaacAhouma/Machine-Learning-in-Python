# import sklearn as sk
import pandas
# import sframe
import json
import numpy as np
from helpers import get_numpy_data,predict_probability,logistic_regression

products = pandas.read_csv('C:\Users\Isaac\Course 3/amazon_baby_subset.csv')

print '# of positive reviews =', len(products[products['sentiment']==1])
print '# of negative reviews =', len(products[products['sentiment']==-1])

# Reading the json file of important words
with open('C:\Users\Isaac\Course 3/important_words.json', 'r') as f:
    important_words = json.load(f)
important_words = [str(s) for s in important_words]

products = products.fillna({'review':''}) 

#Cleaning the review column
def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

products['review_clean'] = products['review'].apply(remove_punctuation)

#Iterating over the words in important words
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))
    
# number of reviews with the word perfect
sum(products['perfect']>0) # 2955

feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment') 

feature_matrix.shape  #193 features

# Computing the optimal coefficients
coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(194),
 step_size=1e-7, max_iter=301)

# Computing the scores for those coefficients
scores = np.dot(feature_matrix, coefficients)

# classifying the scores
negatives = (scores <= 0).astype(int)*(-1) #27946
positives = (scores > 0).astype(int) #25126

predictions = negatives+positives


#Computing accuracy

num_correct = sum(sentiment==predictions)
accuracy = (sum(sentiment==predictions)+0.0)/len(sentiment)
print "-----------------------------------------------------"
print '# Reviews   correctly classified =', num_correct
print '# Reviews incorrectly classified =', len(products) - num_correct
print '# Reviews total                  =', len(products)
print "-----------------------------------------------------"
print 'Accuracy = %.2f' % accuracy

#0.75

# Finding the most positive words
coefficients = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)



