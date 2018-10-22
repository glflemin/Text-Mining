# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:13:00 2018

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

df = pd.read_csv('italy_wine.csv') # what's up with the r's?
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
morewords = ['drink', 'drinks', 'wine', 'wines', 'wine', 'wines', 'grape', 'grapes', 'note', 'notes', 'aroma', 'aromas', 'palate'
             'finish', 'taste', 'tastes', 'show', 'flavour', 'flavor', 'flavors', 'flavours', 'fruit']
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

#empty = " ".join(cln_term_vec[0])
#print(empty)

cln_length = len(cln_term_vec)
empty = []
for i in range(cln_length):
    a = " ".join(cln_term_vec[i])
    empty.append(a)
    
print(empty[0:100])

# wordcloud trial
#text = str(cln_term_vec[0])
#wordcloud = WordCloud().generate(text)
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.show()

# generating one from a pre-saved image
np.set_printoptions(threshold=100000)

# testing; works! 
spain_mask = plt.imread("C:\\Users\\Grant\\Downloads\\italyflag.png")
spain_mask
spain_mask.shape
type(spain_mask)
plt.imshow(spain_mask)
plt.show()

#us_mask # checking color values
spain_mask = np.array(Image.open("C:\\Users\\Grant\\Downloads\\italyflag.png"))
spain_mask.shape
spain_text = str(empty)
spain_wordcloud = WordCloud(background_color="white", mode="RGBA", max_words=500, mask=spain_mask).generate(spain_text) # makes the object
image_colors = ImageColorGenerator(spain_mask)
plt.figure(figsize=[30,30])
plt.imshow(spain_wordcloud.recolor(color_func=image_colors), interpolation="bilinear") # "loads" into plot fx so can display
plt.axis("off") # still not sure what this does
plt.savefig("spain_cloudx.png", format="png")
plt.show()

