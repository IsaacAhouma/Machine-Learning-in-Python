# -*- coding: utf-8 -*-

# Regression Helper methods#

import numpy as np

import sklearn as sk

from math import sqrt

#simple helper to compute residual sum of squares
#Inputs are:
#features : the feature variables of the regression model considered
#response: the response variable
# reg : the fit obtained by doing reg=LinearRegression() . Need to import LinearRegression from sklearn.linear_model
def res_sum_squares(features,response,reg):
    rss=0
    pred=reg.predict(features)
    for i in range(len(pred)):
        residual=pred[i][0]-response[i]
        rss=rss+residual**2
    return rss


# Helper method to compute RSS when fitting model using gradient descent algorithm
# Inputs are the same as those used for the gradient descent method
def res_sum_squares2(features, response,initial_weights, step_size,tolerance):
    rss=0
    weights=regression_gradient_descent(features, response,initial_weights, step_size,tolerance)
    pred=np.dot(features,weights)
    for i in range(len(pred)):
        residual=pred[i]-response[i]
        rss=rss+residual**2
    return rss

# Helper function to obtain the array of features and the response variable variable vector
def get_numpy_data(data, features, output):
    data['constant'] = 1 # add a constant column to an DataFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the ‘features’ list into the SFrame ‘features_sframe’

    # this will convert the features_dataFrame into a numpy matrix
    features_matrix = np.array(data[features])
    # assign the column of data_sframe associated with the target to the variable ‘output_sarray’

    # this will convert the SArray into a numpy array:
    output_array = np.array(data[output])
    return(features_matrix, output_array)

# simple helper to obtain prediction values given a matrix of features and a corresponding vector of weights
def predict(features_matrix,weights):
    predictions=np.dot(features_matrix,weights)
    
    return predictions
    
# simple helper to obtain the derivative of the regression cost function with respect to a given feature
def feature_derivative(errors, feature):
    derivative=2*predict(errors,feature)
    return(derivative)


# Method to find the optimal weights of a model  by gradien descent given 
# feature variables and response(target) variable. Also requires a vector of initial weights, tolerance and a step size.
# outputs a vector of optimal weights(also called coefficients)

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        # compute the errors as predictions - output:
        predictions=predict(feature_matrix,weights)
        errors=predictions-output
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative=feature_derivative(errors,feature_matrix[:,i])
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares=gradient_sum_squares+derivative**2
            
            # update the weight based on step size and derivative:
            weights[i]=weights[i]-derivative*step_size
            
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)