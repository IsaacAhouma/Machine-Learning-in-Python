# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 13:45:37 2017

@author: Isaac
"""

import pandas
import sentiment_analysis
import glob

pos = [type(pandas.read_table(filename)) for filename in glob.glob("tokens/pos/*.txt")]
neg = [pandas.read_table(filename) for filename in glob.glob("tokens/neg/*.txt")]

#pos[0].

temp = glob.glob("tokens/pos/cv003_tok-8338.txt")

mich_train = pandas.read_table("mich_train.txt",names=['sentiment_num','review'])

mich_train['sentiment'] = mich_train['sentiment_num'].apply(lambda sentiment_num : 'Positive' if sentiment_num == 1 else 'Negative')

pred = sentiment_analysis.sentiment_analysis(mich_train['review'])

# accuracy = sum(pred != mich_train['sentiment])/len(pred)