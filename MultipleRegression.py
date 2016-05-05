# -*- coding: utf-8 -*-
# Multiple Linear Regression Script #
import pandas
import numpy as np

from sklearn.linear_model import LinearRegression
from RegressionHelpers import res_sum_squares,get_numpy_data,regression_gradient_descent,res_sum_squares2

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 
'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 
'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales=pandas.read_csv('C:\Users\Isaac\Course 2/kc_house_data.csv')
sales_train=pandas.read_csv('C:\Users\Isaac\Course 2/kc_house_train_data.csv')
sales_test=pandas.read_csv('C:\Users\Isaac\Course 2/kc_house_test_data.csv')

# Adding new variables to training data
sales_train['bedrooms_squared']=sales_train['bedrooms']*sales_train['bedrooms']
sales_train['bed_bath_rooms']=sales_train['bedrooms']*sales_train['bathrooms']
sales_train['log_sqft_living']=np.log(sales_train['sqft_living'])
sales_train['lat_plus_long']=sales_train['lat']+sales_train['long']

# Adding new variables to test data
sales_test['bedrooms_squared']=sales_test['bedrooms']*sales_test['bedrooms']
sales_test['bed_bath_rooms']=sales_test['bedrooms']*sales_test['bathrooms']
sales_test['log_sqft_living']=np.log(sales_test['sqft_living'])
sales_test['lat_plus_long']=sales_test['lat']+sales_test['long']

a,b,c,d=[sales_test['bedrooms_squared'].mean(),
sales_test['bed_bath_rooms'].mean(),
sales_test['log_sqft_living'].mean(),
sales_test['lat_plus_long'].mean()]
#a=12.17,b=7.496,c=7.55,d=-74.65

# Models
target=['price']
model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = model_1_features + ['bed_bath_rooms']
model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']


reg1=LinearRegression()
reg2=LinearRegression()
reg3=LinearRegression()

fit1=reg1.fit(sales_train[model_1_features],sales_train[target])
fit2=reg2.fit(sales_train[model_2_features],sales_train[target])
fit3=reg3.fit(sales_train[model_3_features],sales_train[target])





#Model 1 coefficients and sum of squares residual
fit1.coef_
fit1.intercept_
fit1.coef_[0][2] #15706.742082734609
rss1=res_sum_squares(sales_train[model_1_features],sales_train['price'],reg1) # training data
rss1b=res_sum_squares(sales_test[model_1_features],sales_test['price'],reg1)  # test data




#Model 2 coefficients and sum of squares residual
fit2.coef_
fit2.intercept_
fit2.coef_[0][2] #-71461.308292759204
rss2=res_sum_squares(sales_train[model_2_features],sales_train['price'],reg2) # training data
rss2b=res_sum_squares(sales_test[model_2_features],sales_test['price'],reg2) # test data






#Model 3 coefficients and sum of squares residual
fit3.coef_
fit3.intercept_
rss3=res_sum_squares(sales_train[model_3_features],sales_train['price'],reg3) # training data
rss3b=res_sum_squares(sales_test[model_3_features],sales_test['price'],reg3) # test data

#Model A
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(sales_train, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size,tolerance)
#281.91

(test_matrix1,test_output1)=get_numpy_data(sales_test, simple_features,my_output)

#predicting the sale price of the first house in the dataset using model A
np.dot(simple_weights,np.array([1,sales_test['sqft_living'][0]])) #356134.44325500238


# Residual sum of squares for model A
rss=res_sum_squares2(test_matrix1, test_output1,initial_weights, step_size,tolerance) #275395691278133.28


#rss=np.dot(simple_weights,simple_feature_matrix)

#Model B
model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(sales_train, model_features,my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

(test_matrix2,test_output2)=get_numpy_data(sales_test, model_features,my_output)
multiple_weights=regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)

#predicting the sale price of the first house in the dataset using model B
np.dot(multiple_weights,test_matrix2[0]) #366651.41162949387


# Residual sum of squares for model B
rss2=res_sum_squares2(test_matrix2, test_output2,initial_weights, step_size,tolerance) #269870816068344.53

