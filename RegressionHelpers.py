# -*- coding: utf-8 -*-
# Regression Helper methods#
### Useful for Simple Linear Regression, Multiple Regression, Ridge Regression, Lasso Regression, Nearest Neighbor Regression and Kernel Regression ### 

import numpy as np
import pandas
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.linear_model import LinearRegression

from math import sqrt

#Simple helper Method to compute residual sum of squares
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


#Method to compute RSS when fitting model using gradient descent algorithm
# Inputs are the same as those used for the gradient descent method
def res_sum_squares2(features, response,initial_weights, step_size,tolerance):
    rss=0
    weights=regression_gradient_descent(features, response,initial_weights, step_size,tolerance)
    pred=np.dot(features,weights)
    for i in range(len(pred)):
        residual=pred[i]-response[i]
        rss=rss+residual**2
    return rss

# Method to compute RSS when fitting model using gradient descent algorithm
# Inputs are the same as those used for the gradient descent method
def res_sum_squares_ridge(features, response,initial_weights,penalty, step_size,max_iterations):
    rss=0
    weights=regression_gradient_descent_ridge(features, response,penalty,initial_weights, step_size,max_iterations)
    pred=np.dot(features,weights)
    for i in range(len(pred)):
        residual=pred[i]-response[i]
        rss=rss+residual**2
    return rss

# Helper method to obtain the array of features and the response variable variable vector
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

# Simple helper method to obtain prediction values given a matrix of features and a corresponding vector of weights
def predict(features_matrix,weights):
    predictions=np.dot(features_matrix,weights)
    
    return predictions
    
# Simple helper method to obtain the derivative of the regression cost function with respect to a given feature
def feature_derivative(errors, feature):
    derivative=2*predict(errors,feature)
    return(derivative)

# Simple helper method to obtain the derivative of the regression cost function with respect to a given feature and an l2_penalty used for ridge regression
def feature_derivative_ridge(errors, feature,weight,penalty):
    derivative=2*predict(errors,feature)+2*penalty*weight
    return(derivative)


# Method to find the optimal weights of a model  by gradient descent given 
# feature variables and a response(target) variable. Also requires a vector of initial weights, tolerance and a step size.
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

# Algorithm to find optimal weights of a ridge regression fit via gradient descent.
# Outputs optimal weights
def regression_gradient_descent_ridge(feature_matrix, output,penalty, initial_weights, step_size, max_iterations):
    converged = False
    weights = np.array(initial_weights)
    count=0
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        # compute the errors as predictions - output:
        predictions=predict(feature_matrix,weights)
        errors=predictions-output
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        count=count+1
        for i in range(len(weights)):
            if i == 0:
                derivative=feature_derivative(errors,feature_matrix[:,i])
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            else:
                derivative=feature_derivative_ridge(errors,feature_matrix[:,i],weights[i],penalty)
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares=gradient_sum_squares+derivative**2
            
            # update the weight based on step size and derivative:
            weights[i]=weights[i]-derivative*step_size
            
        gradient_magnitude = sqrt(gradient_sum_squares)
        if max_iterations < count:
            converged = True
    return(weights)

# Creates a data frame where each column is the given feature to the power j. Where j=0,...,degree
def polynomial_dataframe(feature, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pandas.DataFrame()
    temp=['constant','power_1']
    local=['power_1']
    # and set poly_dataframe['power_1'] equal to the passed feature
    poly_dataframe['power_1']=feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name]=feature**power
            temp.append(name)
            local.append(name)
    return poly_dataframe,temp,local

# fit a particular model based on a polynomial dataframe obtained from above helper method (polynomial_dataframe)
def fit_and_plot(data,feature,target,degree):
    poly_data,features,temp=polynomial_dataframe(feature, degree)
    poly_data[target] = data[target]
    regx=LinearRegression()
    (feature_matrix,my_target)=get_numpy_data(poly_data,temp,target)
    model=regx.fit(feature_matrix,my_target)
    #plt.plot(feature_matrix,my_target,'.',
    #poly_data[temp], model.predict(poly_data[features]),'-')
    #return intercept,coefficients,plt
    return model,poly_data,features,temp

# Finds the lowest residual sum of squares for a set of models. Uses polynomial_dataframe() and fit_and_plot()
def find_lowest_rss(training,validation,test,training_feature,validation_feature,test_feature,target,degree):
    min_rss_valid=10**100
    min_rss_test=10**100
    for i in range(degree):
        model,poly_data,features,temp=fit_and_plot(training,training_feature,target,i+1)
        #poly_data,features,temp=polynomial_dataframe(feature, i+1)
        model2,data_validation,features2,temp2=fit_and_plot(validation,validation_feature,target,i+1)
        #data_test,features,temp=polynomial_dataframe(test_feature, i+1)
        model3,data_test,features3,temp3=fit_and_plot(test,test_feature,target,i+1)
        predictions_valid=model.predict(data_validation[features])
        rss_valid=sum((predictions_valid-data_validation[target])**2)
        predictions_test=model.predict(data_test[features])
        rss_test=sum((predictions_test-data_test[target])**2)
        if rss_valid < min_rss_valid:
            min_rss_valid=rss_valid
            lowest_rss_valid=i
            best_model_test_rss=rss_test
            
        if rss_test < min_rss_test:
            min_rss_test = rss_test
            lowest_rss_test=i
    
    return min_rss_valid,lowest_rss_valid,min_rss_test,lowest_rss_test,best_model_test_rss
    
# normalize columns corresponding to given features    
def normalize_features(features):
    norms = np.linalg.norm(features, axis=0)
    normalized_features = features / norms
    return (normalized_features, norms) 


