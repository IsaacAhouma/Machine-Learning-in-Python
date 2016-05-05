
## ensemble of helper methods used to build Logistic classifier with optional L2-regularization
import numpy as np
# Function converting a data frame into a multidimensional array
#Inputs:
#    data_frame: the data frame to be converted
#    features :a list of string containg the label of the columns to be used as features
#    label: a string containing the of the single column that is used as class labels

#Outputs:
#    feature_matrix: 2D array for features
#    label_array: 1D array for class labels

def get_numpy_data(data_frame, features, label):
    data_frame['intercept'] = 1
    features = ['intercept'] + features
    features_frame = data_frame[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = data_frame[label]
    label_array = label_sarray.as_matrix()
    return(feature_matrix, label_array)


# produces probablistic estimate for P(y_i = +1 | x_i, w).
# estimate ranges between 0 and 1.

def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients  
    
    score = np.dot(feature_matrix,coefficients)
    
    # Compute P(y_i = +1 | x_i, w) using the link function
    
    predictions = np.exp(score)/(1+np.exp(score)) # or 1/(1+np.exp(-score))
    
    # return predictions
    return predictions
 
# Output: derivative of the log likelihood with respect to each coefficient w(j)
def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.dot(errors,feature)
    
    # Return the derivative
    return derivative
    
 # Output: derivative of the log likelihood with respect to each coefficient w(j) and the l2-penalty parameter  
def feature_derivative_with_L2(errors, feature, coefficient, l2_penalty, feature_is_constant): 
    
    # Computing the dot product of errors and feature
    
    derivative = np.dot(errors,feature)

    # adding L2 penalty term for any feature that isn't the intercept.
    if not feature_is_constant: 
        derivative = derivative-2*l2_penalty*coefficient
        
    return derivative

# Compute log likelihood of parameters   
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    lp = np.sum((indicator-1)*scores - logexp)
    return lp

# Compute log likelihood of parameters with respect to a given penalty on the l2-norm
def compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, l2_penalty):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    
    lp = np.sum((indicator-1)*scores - np.log(1. + np.exp(-scores))) - l2_penalty*np.sum(coefficients[1:]**2)
    
    return lp
    
from math import sqrt

#  Inputs:
#     feature_matrix: 2D array of features
#     sentiment: 1D array of class labels
#     initial_coefficients: 1D array containing initial values of coefficients
#    step_size: a parameter controlling the size of the gradient steps
#    max_iter: number of iterations to run gradient ascent

# Outputs:
#   Coefficients: The optimal vector of coefficients obtained by gradient ascent

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):

        # Predict P(y_i = +1|x_i,w) using the predict_probability() function
    
        predictions = predict_probability(feature_matrix, coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            
            # feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j].
            
            derivative = feature_derivative(errors, feature_matrix[:,j])
            
            # adding the step size times the derivative to the current coefficient
            coefficients[j]=coefficients[j]+step_size*derivative
        
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients

#  Inputs:
#     feature_matrix: 2D array of features
#     sentiment: 1D array of class labels
#     initial_coefficients: 1D array containing initial values of coefficients
#    step_size: a parameter controlling the size of the gradient steps
#    max_iter: number of iterations to run gradient ascent
#   l2_penalty: Penalty parameter        
# Outputs:
#   Coefficients: The optimal vector of coefficients obtained by gradient ascent with penalty on the l2-norm of the coefficients

def logistic_regression_with_L2(feature_matrix, sentiment, initial_coefficients,l2_penalty, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in xrange(max_iter):

        # Predict P(y_i = +1|x_i,w) using the predict_probability() function
    
        predictions = predict_probability(feature_matrix, coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in xrange(len(coefficients)): # loop over each coefficient
            feature_is_constant=(j==0)
            
            # feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j].
            
            derivative = feature_derivative_with_L2(errors, feature_matrix[:,j],coefficients[j],l2_penalty,feature_is_constant)
            
            # adding the step size times the derivative to the current coefficient
            coefficients[j]=coefficients[j]+step_size*derivative
        
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients,l2_penalty)
            print 'iteration %*d: log likelihood of observed labels = %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients
    

import matplotlib.pyplot as plt
#matplotlib inline
plt.rcParams['figure.figsize'] = 10, 6


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 10, 6

def make_coefficient_plot(table, positive_words, negative_words, l2_penalty_list):
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')
    
    xx = l2_penalty_list
    plt.plot(xx, [0.]*len(xx), '--', lw=1, color='k')
    
    table_positive_words = table[table['word'].isin(positive_words)]
    table_negative_words = table[table['word'].isin(negative_words)]
    del table_positive_words['word']
    del table_negative_words['word']
    
    for i in xrange(len(positive_words)):
        color = cmap_positive(0.8*((i+1)/(len(positive_words)*1.2)+0.15))
        plt.plot(xx, table_positive_words[i:i+1].as_matrix().flatten(),
                 '-', label=positive_words[i], linewidth=4.0, color=color)
        
    for i in xrange(len(negative_words)):
        color = cmap_negative(0.8*((i+1)/(len(negative_words)*1.2)+0.15))
        plt.plot(xx, table_negative_words[i:i+1].as_matrix().flatten(),
                 '-', label=negative_words[i], linewidth=4.0, color=color)
        
    plt.legend(loc='best', ncol=3, prop={'size':16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()

# Outputs: the accuracy of the model on the data
def get_classification_accuracy(feature_matrix, sentiment, coefficients):
    scores = np.dot(feature_matrix, coefficients)
    apply_threshold = np.vectorize(lambda x: 1. if x > 0  else -1.)
    predictions = apply_threshold(scores)
    
    num_correct = float((predictions == sentiment).sum())
    accuracy = num_correct / len(feature_matrix)    
    return accuracy

