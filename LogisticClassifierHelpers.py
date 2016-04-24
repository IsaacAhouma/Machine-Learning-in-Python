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
    
def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.dot(errors,feature)
    
    # Return the derivative
    return derivative
    
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    lp = np.sum((indicator-1)*scores - logexp)
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