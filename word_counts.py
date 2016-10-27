# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 23:24:51 2016

@author: Isaac
"""
import numpy as np

a = 'The quick brown fox jumps over the lazy dog'.lower().split()

b = 'A quick brown dog outpaces a quick fox'.lower().split()


# given a string finds its count in a list of strings

def find_count(string,strings):
    count = 0
    for s in strings:
        if s==string:
            count += 1
    return count
    
# Given 2 lists of strings, compute their Cosine distance. Cosine distance is equal to 1 - Cosine Similarity
def cosine_distance(string1,string2):
    res1 = []
    res2 = []
    acc = []
    str_array2 = []
    str_array1 = []
    for string in string1+string2:
        if string not in acc:
            count1 = find_count(string,string1)
            count2 = find_count(string,string2)
            acc.append(string)
            res1.append(count1)
            res2.append(count2)
            str_array1.append(string)
            str_array2.append(string)
    return str_array1,str_array2,1-float(np.dot(res1,res2))/(np.linalg.norm(res1)*np.linalg.norm(res2))

# Given 2 lists of strings, compute their Euclidean distance  
def euclidean_distance(string1,string2):
    res1 = []
    res2 = []
    acc = []
    str_array1 = []
    str_array2 = []
    for string in string1+string2:
        if string not in acc:
            count1 = find_count(string,string1)
            count2 = find_count(string,string2)
            acc.append(string)
            res1.append(count1)
            res2.append(count2)
            str_array1.append(string)
            str_array2.append(string)
    return str_array1,str_array2,np.linalg.norm(np.array(res1)-np.array(res2))

print cosine_distance(a,b)
print euclidean_distance(a,b)
    
    
    