'''
Created on Tue Oct  9 22:52:59 2018

@author: Grant, Pierce
'''
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

# Import US wine reviews as a dataframe, create a list from the dataframe
df = pd.read_csv(r'Wine_Review_US.csv') # what's up with the r's?
doc = []
descriptions = df['description']
type(descriptions) # made a series

df.head()
for description in descriptions:
    doc.append(description)
    
# remove punctuation from all reviews
#nltk.download('punkt')
punc = re.compile('[%s]' % re.escape(string.punctuation))
term_vec = []
for d in doc: 
    d = d.lower()
    d = punc.sub('', d)
    term_vec.append(d)

print(term_vec[0:10])

#tokenize the text in each of the reviews
token_term_vec=[]
for elm in term_vec:
    token_term_vec.append(nltk.word_tokenize(elm))
    
print(token_term_vec[0:10]) # looks good

# add words that might be relevant stop words
stopwords = nltk.corpus.stopwords.words('english')
morewords = ['drink', 'drinks', 'wine', 'wines', 'wine', 'wines', 'grape', 'grapes', 'note', 'notes', 'aroma', 'palate'
             'finish', 'taste', 'tastes', 'show', 'flavour', 'flavor', 'flavors', 'flavours', 'fruit']
stopwords.extend(morewords)

print(stopwords)

#remove stop words from the reviews
nsw_term_vec = []
for review in token_term_vec: 
    wr=[]
    for word in review:
        if word not in stopwords:
            wr.append(word)    
    nsw_term_vec.append(wr)
            
print(nsw_term_vec[0:10]) # looks good!

# Lemmatize data to combine singular and plural word forms
wnl = nltk.stem.WordNetLemmatizer()

cln_term_vec = []
for reviews in nsw_term_vec:
    lemma=[]
    for words in reviews:
        lemma.append(wnl.lemmatize(words))
    cln_term_vec.append(lemma)
    
print(cln_term_vec[0:10]) # looks good!

#empty = " ".join(cln_term_vec[0])
#print(empty)

# Final list of lists to put into word cloud and do analysis
cln_length = len(cln_term_vec)
empty = []
for i in range(cln_length):
    a = " ".join(cln_term_vec[i])
    empty.append(a)
    
print(empty[0:10]) # looks good!


# Wordcloud time
###############################################################################

# example mask 
wine_mask = np.array(Image.open("C:\\Users\\Grant\\Downloads\\wine_mask.png"))
plt.imshow(wine_mask)
plt.show()

np.set_printoptions(threshold=1000)

wine_mask[wine_mask == 0] = 255
wine_mask
plt.imshow(wine_mask)
plt.show()
wc = WordCloud(background_color="white", max_words=1000, mask=wine_mask,
               stopwords=stopwords, contour_width=3, contour_color='firebrick')

# generating a mask from a pre-saved image
us_mask = np.array(Image.open("C:\\Users\\Grant\\Downloads\\usflag2.png"))
#us_mask # checking color values
plt.imshow(us_mask)
plt.show()

# now to generate a wordcloud
us_text = str(empty)
us_wordcloud = WordCloud(background_color="white", mode="RGBA", max_words=500, mask=us_mask).generate(us_text) # makes the object
image_colors = ImageColorGenerator(us_mask)
plt.figure(figsize=[30,30])
plt.imshow(us_wordcloud.recolor(color_func=image_colors), interpolation="bilinear") # "loads" into plot fx so can display
plt.axis("off") # still not sure what this does
plt.savefig("us_cloud.png", format="png")

# now trying an image of the U.S as a country
us2_mask = np.array(Image.open("C:\\Users\\Grant\\Downloads\\usoutline.png")) # transparent version is bad. Gotta change
us2_mask
plt.imshow(us2_mask)
plt.show()

us2_mask[us2_mask == 0] = 255
plt.imshow(us2_mask)
plt.show()

us_text = str(empty)
us2_wordcloud = WordCloud(background_color="white", max_words=500, mask=us2_mask, contour_width=2, contour_color='firebrick') # makes the object
us2_wordcloud.generate(us_text)
us2_wordcloud.to_file("hateverything.png")
plt.figure(figsize=[50, 50])
plt.axis("off")
plt.imshow(us2_wordcloud, interpolation="bilinear") # "loads" into plot fx so can display # still not sure what this does
plt.show()

# Everything below is WIP. Code takes forever to run! 
#########################################################################

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



    