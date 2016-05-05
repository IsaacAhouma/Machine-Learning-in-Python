## Script used to analyze sentiment
# import sklearn as sk
import pandas
# import sframe
import json
import numpy as np
from LogisticClassifierHelpers import get_numpy_data,logistic_regression,
logistic_regression_with_L2,make_coefficient_plot

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



#



with open('C:\Users\Isaac\Course 3/module-4-assignment-train-idx.json', 'r') as f:
    hold1 = json.load(f)
hold1 = [int(i) for i in hold1]

with open('C:\Users\Isaac\Course 3/module-4-assignment-validation-idx.json', 'r') as f:
    hold2 = json.load(f)
hold2 = [int(i) for i in hold2]

train_data=products.loc[hold1]
validation_data=products.loc[hold2]
feature_matrix_train, sentiment_train = get_numpy_data(train_data, important_words, 'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(validation_data, important_words, 'sentiment') 


# Analysis of Benefits of using L2-regularization
l21=0
l22=4
l23=10
l24=1e2
l25=1e3
l26=1e5
step_size=5e-6
max_iter=501
initial_coefficients=np.zeros(194)

coefficients_0_penalty= logistic_regression_with_L2(feature_matrix_train, sentiment_train, initial_coefficients,l21, step_size, max_iter)
coefficients_4_penalty= logistic_regression_with_L2(feature_matrix_train, sentiment_train, initial_coefficients,l22, step_size, max_iter)
coefficients_10_penalty= logistic_regression_with_L2(feature_matrix_train, sentiment_train, initial_coefficients,l23, step_size, max_iter)
coefficients_1e2_penalty= logistic_regression_with_L2(feature_matrix_train, sentiment_train, initial_coefficients,l24, step_size, max_iter)
coefficients_1e3_penalty= logistic_regression_with_L2(feature_matrix_train, sentiment_train, initial_coefficients,l25, step_size, max_iter)
coefficients_1e5_penalty= logistic_regression_with_L2(feature_matrix_train, sentiment_train, initial_coefficients,l26, step_size, max_iter)

top_5=np.sort(coefficients_0_penalty)[-5:]

bottom_5=np.sort(coefficients_0_penalty)[:5]

table = pandas.DataFrame({'word': ['(intercept)'] + important_words})
def add_coefficients_to_table(coefficients, column_name):
    table[column_name] = coefficients
    return table

add_coefficients_to_table(coefficients_0_penalty, 'coefficients [L2=0]')
add_coefficients_to_table(coefficients_4_penalty, 'coefficients [L2=4]')
add_coefficients_to_table(coefficients_10_penalty, 'coefficients [L2=10]')
add_coefficients_to_table(coefficients_1e2_penalty, 'coefficients [L2=1e2]')
add_coefficients_to_table(coefficients_1e3_penalty, 'coefficients [L2=1e3]')
add_coefficients_to_table(coefficients_1e5_penalty, 'coefficients [L2=1e5]')

table[table['coefficients [L2=0]']==top_5]

sorted_table=table.sort_values('coefficients [L2=0]')

positive_words=sorted_table[-5:]['word'].tolist()
negative_words=sorted_table[:5]['word'].tolist()
ind_neg=[106,97,114,113,169]
ind_pos=[3,34,8,23,4]

make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list=[0, 4, 10, 1e2, 1e3, 1e5])

train_accuracy = {}
train_accuracy[0]   = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_0_penalty)
train_accuracy[4]   = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_4_penalty)
train_accuracy[10]  = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_10_penalty)
train_accuracy[1e2] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_1e2_penalty)
train_accuracy[1e3] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_1e3_penalty)
train_accuracy[1e5] = get_classification_accuracy(feature_matrix_train, sentiment_train, coefficients_1e5_penalty)

validation_accuracy = {}
validation_accuracy[0]   = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_0_penalty)
validation_accuracy[4]   = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_4_penalty)
validation_accuracy[10]  = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_10_penalty)
validation_accuracy[1e2] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_1e2_penalty)
validation_accuracy[1e3] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_1e3_penalty)
validation_accuracy[1e5] = get_classification_accuracy(feature_matrix_valid, sentiment_valid, coefficients_1e5_penalty)

# Build a simple report
for key in sorted(validation_accuracy.keys()):
    print "L2 penalty = %g" % key
    print "train accuracy = %s, validation_accuracy = %s" % (train_accuracy[key], validation_accuracy[key])
    print "--------------------------------------------------------------------------------"
