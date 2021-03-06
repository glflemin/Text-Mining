'''
Created on Tue Oct  9 22:52:59 2018

@author: Grant, Pierce
'''


import os
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
import re
import scipy 
import string
import sklearn as sl 
import pandas as pd
import gensim as gs
import pickle as pkl
import PIL 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

os.chdir(r"C:\\Users\\pseco\\Documents\\GitHub\\Text-Mining\\") 
#os.chdir("C:\\Users\\Grant\\Documents\\GitHub\\Text-Mining\\")

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

print(term_vec[0:10])

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

# wordcloud trial
text = str(cln_term_vec[0])
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

print(text)
#Gen Sim Dictionary
    
dict = gs.corpora.Dictionary(cln_term_vec)

corpora=[]
for i in range(0, len(cln_term_vec)):
    corpora.append(dict.doc2bow(cln_term_vec[i]))


#Create TFIDF Vectors Based on term Vectors

tfidf_model = gs.models.TfidfModel(corpora)

tfidf=[]
for i in range(0, len(corpora)):
    tfidf.append(tfidf_model[corpora[i]])
    
#Create Pairwise Document Siliarity Index
n=len(dict)
index = gs.similarities.SparseMatrixSimilarity(tfidf_model[corpora], num_features = n)    

#Prints Words with their respective index

print(dict.token2id)

#TFIDF Values per Document

for i in range(0,len(tfidf)):
    s='Review' + ' ' + str(i+1)+' TFIDF'
    
    for j in range(0, len(tfidf[i])):
        s = s + ' (' + dict.get(tfidf[i][j][0]) + ',' 
        s = s + ('%.3f' % tfidf[i][j][1]) + ')'
               
    print(s)
    
# Produces Document Simalarity
#rev_sim = []
for i in range(0, len(corpora)):
    sim = index[tfidf_model[corpora[i]]]
    for j in range(0, len(sim)):
        #rev_sim.append(sim[j])
        sim[j]

print(index[tfidf_model[corpora[0]]][1])

    