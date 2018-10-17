'''
Created on Tue Oct  9 22:52:59 2018

@author: Grant, Pierce
'''


import os
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
import re
import string
import sklearn as sl 
import pandas as pd
import gensim as gs
import pickle as pkl

os.chdir(r"C:\\Users\\pseco\\Documents\\GitHub\\Text-Mining\\") 
#os.chdir("C:\\Users\\gflemin\\Documents\\GitHub\\Text-Mining\\")

df = pd.read_csv(r'Wine_Review_US.csv') # what's up with the r's?
doc = []
descriptions = df['description']
type(descriptions) # made a series

df.head()
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

#print(term_vec[0:10])

#tokenize
token_term_vec=[]
for elm in term_vec:
    token_term_vec.append(nltk.word_tokenize(elm))
    
print(token_term_vec[0:10])

# add words that might be relevant stop words
stopwords = nltk.corpus.stopwords.words('english')
morewords = ['drink', 'drinks', 'wine', 'wines', 'note', 'notes', 'flavor', 'flavors',
             'wine', 'wines', 'grape', 'grapes']
stopwords.extend(morewords)

#print(stopwords)

#remove stop words
#nltk.download('stopwords')
nsw_term_vec = []
for review in token_term_vec: 
    wr=[]
    for word in review:
        if word not in stopwords:
            wr.append(word)    
    nsw_term_vec.append(wr)
            
print(nsw_term_vec[0:10])

# Lemmatize data
#nltk.download('wordnet')
wnl = nltk.stem.WordNetLemmatizer()

cln_term_vec = []
for reviews in nsw_term_vec:
    lemma=[]
    for words in reviews:
        lemma.append(wnl.lemmatize(words))
    cln_term_vec.append(lemma)
    
print(cln_term_vec[0:10])

#Gen Sim Dictionary
    
dict = gs.corpora.Dictionary(cln_term_vec)

corp=[]
for i in range(len(cln_term_vec)):
    corp = corp + dict.doc2bow(cln_term_vec[i])

path = open('Wine_Txt_Corpus_Dict.csv', 'wb')
pkl.dump(corp, path)
path.close()


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
    print('Review', (i+1), 'sim: [ '),
    
    sim = index[tfidf_model[corp[i]]]
    for j in range(0, len(sim)):
        print('%.3f ' %sim[j]),

    print(']')


    