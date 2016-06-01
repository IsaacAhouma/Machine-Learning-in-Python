# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import graphlab
import DecisionTreeHelpers

loans = graphlab.SFrame('lending-club-data.gl/')
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
loans = loans[features + [target]]
safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

# Since there are less risky loans than safe loans, find the ratio of the sizes
# and use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)

loans_data = risky_loans.append(safe_loans)
for feature in features:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})    
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)
    
    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)


features = loans_data.column_names()
features.remove('safe_loans')  # Remove the response variable
features

print "Number of features (after binarizing categorical variables) = %s" % len(features)

loans_data['grade.A']

print "Total number of grade.A loans : %s" % loans_data['grade.A'].sum()
print "Expected answer               : 6422"

train_data, test_data = loans_data.random_split(.8, seed=1)
###################################################################
my_decision_tree=DecisionTreeHelpers.decision_tree_create(train_data,features,'safe_loans',max_depth=6)

test_data[0]

print 'Predicted class: %s ' % DecisionTreeHelpers.classify(my_decision_tree, test_data[0])

DecisionTreeHelpers.classify(my_decision_tree, test_data[0], annotate=True)

DecisionTreeHelpers.evaluate_classification_error(my_decision_tree, test_data)

DecisionTreeHelpers.print_stump(my_decision_tree)

DecisionTreeHelpers.print_stump(my_decision_tree['left']['left'], my_decision_tree['left']['splitting_feature'])