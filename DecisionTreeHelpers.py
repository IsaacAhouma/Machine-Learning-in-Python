
"""
Created on Wed Jun 01 12:49:25 2016

@author: Isaac
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 31 10:30:20 2016

@author: Isaac

"""
import graphlab
# Method 1
#target='safe_loans'
def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    # Count the number of 1's (safe loans)
    count_pos=0
    for i in range(len(labels_in_node)):
        if labels_in_node[i]==1:
            count_pos+=1    
    # Count the number of -1's (risky loans)
    count_neg=0
    for i in range(len(labels_in_node)):
        if labels_in_node[i]==-1:
            count_neg+=1                
    # Return the number of mistakes that the majority classifier makes.
    if count_pos > count_neg:
        majority_class=count_pos
        minority_class=count_neg
    else:
        majority_class=count_neg
        minority_class=count_pos      
    return minority_class

def best_splitting_feature(data, features, target):
    
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))     
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        # The right split will have all data points where the feature value is 1
        right_split =  data[data[feature] == 1]
        # Calculate the number of misclassified examples in the left split.
        left_mistakes = intermediate_node_num_mistakes(left_split[target])           
        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[target])         
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error = (left_mistakes+right_mistakes)/num_data_points
        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        if error < best_error:
            best_error=error
            best_feature=feature 
    return best_feature # Return best feature

def create_leaf(target_values): 
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True    }
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1     
    # Return the leaf node        
    return leaf
    
def decision_tree_create(data, features, target, current_depth = 0, 
                         max_depth = 10, min_node_size=1, 
                         min_error_reduction=0.0):
    
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
    
    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached. All data points have the same target value."                
        return create_leaf(target_values)
    
    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print "Stopping condition 2 reached. No remaining features."                
        return create_leaf(target_values)    
    
    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print "Early stopping condition 1 reached. Reached maximum depth."
        return create_leaf(target_values)
    
    # Early stopping condition 2: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if reached_minimum_node_size(data, min_node_size) is True: ## YOUR CODE HERE 
        print "Early stopping condition 2 reached. Reached minimum node size."
        return create_leaf(target_values)  ## YOUR CODE HERE
    
    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples 
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))
    
    # Calculate the error after splitting (number of misclassified examples 
    # in both groups divided by the total number of examples)
    left_mistakes = intermediate_node_num_mistakes(left_split[target])   ## YOUR CODE HERE
    right_mistakes = intermediate_node_num_mistakes(right_split[target])  ## YOUR CODE HERE
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))
    
    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:        ## YOUR CODE HERE
        print "Early stopping condition 3 reached. Minimum error reduction."
        return create_leaf(target_values)  ## YOUR CODE HERE 
    
    
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split))
    
    
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, 
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)        
    
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, 
                                      current_depth + 1, max_depth, min_node_size, min_error_reduction)        
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}


def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])

def classify(tree, x, annotate = False):   
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction'] 
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x)) 
    #calculate the classification error and return it
    return sum(prediction==data['safe_loans'])/float(len(data))


def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))


def reached_minimum_node_size(data, min_node_size):
    return data <= min_node_size

def error_reduction(error_before_split, error_after_split):
    return error_before_split-error_after_split

def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])
