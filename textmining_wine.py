# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 22:52:59 2018

@author: Grant, Pierce

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
import gensim as gs

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

for review in token_term_vec: 
    wr=[]
    for word in review:
        if word not in stop_words:
            wr.append(word)    
    nsw_term_vec.append(wr)
            

# Lemmatize data
#nltk.download('wordnet')
wln = nltk.stem.WordNetLemmatizer()

cln_term_vec = []
for reviews in nsw_term_vec:
    lemma=[]
    for words in reviews:
        lemma.append(wln.lemmatize(words))
    cln_term_vec.append(lemma)
    
#Gen Sim Dictionary
    
dict = gs.corpora.Dictionary(cln_term_vec)

corp=[]
for i in range(0, len(cln_term_vec)):
    corp.append(dict.doc2bow(cln_term_vec[i]))

#Create TFIDF Vectors Based on term Vectors

tfidf_model = gs.models.TfidfModel(corp)

tfidf=[]
for i in range(0,len(corp)):
    tfidf.append(tfidf_model[corp[i]])
    
#Create Pairwise Document Siliarity Index
n=len(dict)
index = gs.similarities.SparseMatrixSimilarity(tfidf_model[corp], num_features = n)    
    
#Create Pairwise Similarity Per Review

for i in range(0,len(tfidf)):
    s='Review' + str(i+1)+' TFIDF'
    
    for j in range(0, len(tfidf[i])):
        s = s + ' (' + dict.get(tfidf[i][j][0]) + ',' 
        s = s + ('%.3f' % tfidf[i][j][1]) + ')'
               
    print(s)
    
for i in range(0, len(corp)):
    print( 'Review', (i+1), 'sim: [ ',
    
    sim = index[tfidf_model[corp[i]]]
    for j in range(0, len(sim)):
        print('%.3f ' %sim[j]),

    print(']'))


    


