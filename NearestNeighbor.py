import pandas as pd
from sklearn import linear_model  # using scikit-learn
import numpy as np
from RegressionHelpers import predict,get_numpy_data,normalize_features

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int,
'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 
'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data_small.csv', dtype=dtype_dict)
train = pd.read_csv('kc_house_data_small_train.csv', dtype=dtype_dict)
validation = pd.read_csv('kc_house_data_validation.csv', dtype=dtype_dict)
test = pd.read_csv('kc_house_data_small_test.csv', dtype=dtype_dict)

feature_list = ['bedrooms',  
                'bathrooms',  
                'sqft_living',  
                'sqft_lot',  
                'floors',
                'waterfront',  
                'view',  
                'condition',  
                'grade',  
                'sqft_above',  
                'sqft_basement',
                'yr_built',  
                'yr_renovated',  
                'lat',  
                'long',  
                'sqft_living15',  
                'sqft_lot15']
                
features_train, output_train = get_numpy_data(train, feature_list, 'price')
features_test, output_test = get_numpy_data(test, feature_list, 'price')
features_valid, output_valid = get_numpy_data(validation, feature_list, 'price')

features_train, norms = normalize_features(features_train) # normalize training set features (columns)
features_test = features_test / norms # normalize test set by training set norms
features_valid = features_valid / norms # normalize validation set by training set norms

print features_test[0]
print features_train[9]

dist1=np.sqrt(np.sum((features_test[0]-features_test[9])**2)) #dist1==0.058352853644336386


distances=[]
smallest_dist=1000000
for i in range(len(features_train[0:10])):
    dist=np.sqrt(np.sum((features_test[0]-features_train[i])**2))
    distances.append(dist)
    if dist < smallest_dist:
        smallest_dist=dist
        closest_house=i
# closest==8
for i in xrange(3):
    print features_train[i]-features_test[0]
    # should print 3 vectors of length 18

diff=features_train-features_test[0]

alldistances=np.sqrt(np.sum(diff**2, axis=1))

# Compute the Euclidean distance between two observations(ie, how similar they are)
def compute_distances(features_instances, features_query):
    
    diff=features_instances-features_query
    distances=np.sqrt(np.sum(diff**2, axis=1))
    nearest_neighbor=np.argmin(distances)
    return distances,nearest_neighbor

(my_distances,nearest_neighbor)=compute_distances(features_train, features_test[2]) #closest==382

predicted_price=train['price'][closest_house] # predicted_price==249000.0

# Find the k closest (most similar) observations to a given query
def k_nearest_neighbors(k, feature_train, features_query):
    distances,temp=compute_distances(features_train, features_query)
    rank=np.argsort(distances)
    nearest_neighbors=rank[0:k]
    return nearest_neighbors

nearest_neighbors=k_nearest_neighbors(4,features_train, features_test[2])
#[ 382, 1149, 4087, 3142]

# Make a prediction of the output value for a given query observation
def predict_output_of_query(k, features_train, output_train, features_query):
    nearest_neighbors=k_nearest_neighbors(k, features_train, features_query)
    indices=[]
    for i in range(len(nearest_neighbors)):
        indices.append(nearest_neighbors[i])
    prediction=np.mean(output_train[indices])
    
    return prediction

my_prediction=predict_output_of_query(4,features_train,train['price'], features_test[2])
#413987.5

# Predict all the output values for a given set of query observations
def predict_output(k, features_train, output_train, features_query):
    predictions=[]
    for i in range(len(features_query)):
        prediction=predict_output_of_query(k, features_train, output_train, features_query[i])
        predictions.append(prediction)
    return predictions

my_predictions=predict_output(10,features_train,train['price'], features_test[0:10])
#[881300.0,
# 431860.0,
# 460595.0,
# 430200.0,
# 766750.0,
# 667420.0,
# 350032.0,
# 512800.70000000001,
# 484000.0,
# 457235.0]

# Find the best k value using validation set
def find_min_rss_knn(k_values, features_train, output_train, features_query,features_output):
    min_rss=10**100
    for i in range(len(k_values)):
        predictions=predict_output(k_values[i], features_train, output_train, features_query)
        rss=sum((predictions-features_output)**2)
        if rss < min_rss:
            min_rss=rss
            opt_k=k_values[i]
    return opt_k,min_rss

k_values=np.arange(1,16)
opt_k,min_rss=find_min_rss_knn(k_values, features_train, output_train, features_valid, output_valid)
#8
