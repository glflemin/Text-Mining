# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 22:52:59 2018

@author: Grant
"""

import os
import nltk
import re
import string
import sklearn as sl 
import pandas as pd

os.chdir("C:\\Users\\Grant\\Downloads\\wine-reviews\\") 

df = pd.read_csv('winemag-data_first150k.csv', index_col='id')
doc = []
descriptions = df['description']
type(descriptions) # made a series

for description in descriptions:
    doc.append(description)
    
# remove punctuation
punc = re.compile('[%s]' % re.escape(string.punctuation))
term_vec = []
for d in doc: 
    d = d.lower()
    d = punc.sub('', d)
    term_vec.append(nltk.word_tokenize(d))

print(term_vec[1:10])


