# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:44:10 2018

@author: Grant
"""

# SETUP
##################################################################
import os
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
import re
import scipy 
import string
import numpy as np
import sklearn as sl 
import pandas as pd
import gensim as gs
import pickle as pkl
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

#os.chdir(r"C:\\Users\\pseco\\Documents\\GitHub\\Text-Mining\\") 
os.chdir("C:\\Users\\Grant\\Documents\\GitHub\\Text-Mining\\")

###################################################################

df = pd.read_csv('spain_wine.csv') # what's up with the r's?
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
    
print(token_term_vec[0])

# add words that might be relevant stop words
stopwords = nltk.corpus.stopwords.words('english')
morewords = ['drink', 'drinks', 'wine', 'wines', 'wine', 'wines', 'grape', 'grapes', 'note', 'notes', 'aroma', 'aromas', 'palate'
             'finish', 'taste', 'tastes', 'show', 'flavour', 'flavor', 'flavors', 'flavours', 'fruit']
stopwords.extend(morewords)

print(stopwords)

#remove stop words
#nltk.download('stopwords')
nsw_term_vec = []
for review in token_term_vec: 
    wr=[]
    for word in review:
        if word not in stopwords:
            wr.append(word)    
    nsw_term_vec.append(wr)
            
print(nsw_term_vec[0])

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

#empty = " ".join(cln_term_vec[0])
#print(empty)

cln_length = len(cln_term_vec)
empty = []
for i in range(cln_length):
    a = " ".join(cln_term_vec[i])
    empty.append(a)
    
print(empty[0:100])

# testing works! 
spain_mask = plt.imread("C:\\Users\\Grant\\Downloads\\spainflag.png")
#spain_mask
spain_mask.shape
type(spain_mask)
plt.imshow(spain_mask)
plt.show()

#us_mask # checking color values
spain_mask = np.array(Image.open("C:\\Users\\Grant\\Downloads\\spainflag.png"))
spain_text = str(empty)
spain_wordcloud = WordCloud(background_color="white", max_words=500, mask=spain_mask).generate(spain_text) # makes the object
image_colors = ImageColorGenerator(spain_mask)
plt.figure(figsize=[30,30])
plt.imshow(spain_wordcloud.recolor(color_func=image_colors), interpolation="bilinear") # "loads" into plot fx so can display
plt.axis("off") # still not sure what this does
plt.savefig("spain_cloud.png", format="png")
plt.show()

# Now trying an image of Spain as a country
spain2_mask = np.array(Image.open("C:\\Users\\Grant\\Downloads\\spainmap.png")) # transparent version is bad. Gotta change
spain2_mask.shape
spain2_mask
plt.imshow(spain2_mask)
plt.show()

spain2_mask[spain2_mask != 255] = 0
spain2_mask.shape
plt.imshow(spain2_mask)
plt.show()


us_text = str(empty)
us2_wordcloud = WordCloud(background_color="white", max_words=500, mask=spain2_mask, contour_width=2, contour_color='firebrick') # makes the object
us2_wordcloud.generate(us_text)
us2_wordcloud.to_file("hateverything.png")
plt.figure(figsize=[50, 50])
plt.axis("off")
plt.imshow(us2_wordcloud, interpolation="bilinear") # "loads" into plot fx so can display # still not sure what this does
plt.show()

#Gen Sim Dictionary
    
dict = gs.corpora.Dictionary(cln_term_vec)

corpora=[]
for i in range(0, len(cln_term_vec)):
    corpora.append(dict.doc2bow(cln_term_vec[i]))

path = open('Wine_Txt_Corpus_Dict.csv', 'wb')
pkl.dump(corpora, path)
path.close()


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

#Create Pairwise Similarity Per Review

for i in range(0,len(tfidf)):
    s='Review' + ' ' + str(i+1)+' TFIDF'
    
    for j in range(0, len(tfidf[i])):
        s = s + ' (' + dict.get(tfidf[i][j][0]) + ',' 
        s = s + ('%.3f' % tfidf[i][j][1]) + ')'
               
    print(s)
    
for i in range(0, len(corpora)):
    print('Review', (i+1), 'sim: [ '),
    
    sim = index[tfidf_model[corpora[i]]]
    for j in range(0, len(sim)):
        print('%.3f ' %sim[j]),

    print(']')



    