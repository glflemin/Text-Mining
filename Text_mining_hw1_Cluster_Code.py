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


variety = list(df['variety'])
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

corpus=[]
for i in range(0, len(cln_term_vec)):
    corpus.append(dict.doc2bow(cln_term_vec[i]))



#Create TFIDF Vectors Based on term Vectors

tfidf_model = gs.models.TfidfModel(corpus)

tfidf=[]
for i in range(0,len(corpus)):
    tfidf.append(tfidf_model[corpus[i]])
    
#Create Pairwise Document Siliarity Index
n=len(dict)
index = gs.similarities.SparseMatrixSimilarity(tfidf_model[corpus], num_features = n)    

#Prints Words with their respective index

print(dict.token2id)

#TFIDF Values per Document

'''
for i in range(0,len(tfidf)):
    s='Review' + ' ' + str(i+1)+' TFIDF'
    
    for j in range(0, len(tfidf[i])):
        s = s + ' (' + dict.get(tfidf[i][j][0]) + ',' 
        s = s + ('%.3f' % tfidf[i][j][1]) + ')'
'''
    
# Produces Document Simalarity

#Repeat Process stem and tokenize
for i in range(0, len(corpus)):
    sim = index[tfidf_model[corpus[i]]]
    for j in range(0, len(sim)):
        sim[j]    

#cluster analysis
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [wnl.lemmatize(t) for t in filtered_tokens]
    return stems 

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in doc:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
#Define Vectorizer Parameters

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np

#computes similarity
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, stop_words=stopwords, use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(doc)

#Subset Matrix and compute distance
part_one = tfidf_matrix[0:5000]
part_two = tfidf_matrix[5001:10000]
part_three  = tfidf_matrix[10001:15000]
part_four = tfidf_matrix[0:5000]
part_five = tfidf_matrix[5001:10000]
part_six = tfidf_matrix[10001:15000]
dist_matrix_one = 1-cosine_similarity(part_one)
dist_matrix_two = 1-cosine_similarity(part_two)
dist_matrix_three = 1-cosine_similarity(part_three)


terms= tfidf_vectorizer.get_feature_names()

#Cluster Analysis
num_clusters = 50

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters=km.labels_.tolist()

km=joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

#Visualize Cluster
reviews = {'Wine':variety, 'Reviews':doc,'cluster':clusters}

cluster_frame=pd.DataFrame(reviews, index = [clusters], columns = ['Wine', 'cluster'])

print(cluster_frame[0:10])

cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#subset matrix
                 
#create data frame that has the result of the MDS plus the cluster numbers and titles

import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed",  random_state=1)

pos = mds.fit_transform(dist_matrix_one)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()

df_2 = pd.DataFrame(dict(x=xs, y=ys, label=clusters[0:5000], title=variety[0:5000])) 

#group by cluster
groups = df_2.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=clusters, color=clusters
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelleft=False)

for i in range(len(df_2)):
    ax.text(df_2.loc[i]['x'], df_2.loc[i]['y'], df_2.loc[i]['title'], size=8)  

    
    
plt.show() #show the plot

    