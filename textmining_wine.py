# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 22:52:59 2018

@author: Grant, Pierce
"""


"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 11:05:03 2018

@author: pseco
"""

import os
import nltk
import re
import string
import sklearn as sl 
import pandas as pd

os.chdir(r"C:\\Users\\pseco\\Downloads\\") 

df = pd.read_csv(r'winemag-data_first150k.csv')
doc = []
descriptions = df['description']
type(descriptions) # made a series

for description in descriptions:
    doc.append(description)
    
# remove punctuation
#nltk.download('punkt')
punc = re.compile('[%s]' % re.escape(string.punctuation))
term_vec = []
for d in doc: 
    d = d.lower()
    d = punc.sub('', d)
    term_vec.append(d)

token_term_vec=[]
for elm in term_vec:
    token_term_vec.append(nltk.word_tokenize(elm))

#remove stop words
#nltk.download('stopwords')
stop_words= set(nltk.corpus.stopwords.words('english'))


nsw_term_vec = []
for reviews in token_term_vec:
    for words in reviews:
        if words not in stop_words:
            nsw_term_vec.append(words)

# Lemmatize and tokenize data
#nltk.download('wordnet')
wln = nltk.stem.WordNetLemmatizer()

cln_term_vec = []
for words in nsw_term_vec:
        cln_term_vec.append(wln.lemmatize(words))
