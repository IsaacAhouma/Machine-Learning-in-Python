import pandas as ps
import json
import sklearn as sk
import sklearn.tree
import numpy as np
import regression
from LogisticClassifierHelpers import get_numpy_data
loans = ps.read_csv('lending-club-data.csv')
loans.column()
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans',1)

def percentage(column):
    count =0
    for i in range(len(column)):
        if column[i]==1:
            count+=1
    pct = (count+0.0) / len(column)
    
    return pct

percentage(loans['safe_loans']) #0.8111853319957262


features = ['grade',                     # grade of the loan
            'sub_grade',                 # sub-grade of the loan
            'short_emp',                 # one year or less of employment
            'emp_length_num',            # number of years of employment
            'home_ownership',            # home_ownership status: own, mortgage or rent
            'dti',                       # debt to income ratio
            'purpose',                   # the purpose of the loan
            'term',                      # the term of the loan
            'last_delinq_none',          # has borrower had a delinquincy
            'last_major_derog_none',     # has borrower had 90 day or worse rating
            'revol_util',                # percent of available credit being used
            'total_rec_late_fee',        # total late fees received to day
           ]

target = 'safe_loans'

loans = loans[features + [target]]


#train_idx=ps.read_json('module-5-assignment-1-train-idx.json')
with open('C:\Users\Isaac\Course 3/module-5-assignment-1-train-idx.json', 'r') as f:
    train_idx = json.load(f)

#test_idx=ps.read_json('module-5-assignment-1-test-idx.json')
with open('C:\Users\Isaac\Course 3/module-5-assignment-1-validation-idx.json', 'r') as f:
    validation_idx = json.load(f)
    
train_data = loans.iloc[train_idx]
validation_data = loans.iloc[validation_idx]
train_matrix,train_output=get_numpy_data(train_data,features,target)
validation_matrix,validation_output=get_numpy_data(validation_data,features,target)

safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print "Number of safe loans  : %s" % len(safe_loans_raw)
print "Number of risky loans : %s" % len(risky_loans_raw)
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(percentage, seed=1)

tree=sklearn.tree.DecisionTreeClassifier(max_depth=6)

decision_tree_model=tree.fit(train_matrix,train_output)