import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#Libraries permitted to Use
import pandas as pd

import gensim 
from gensim.models import Word2Vec 

#standard python libraries
import math
from collections import defaultdict

#This is just to perform the train test split on the dataset, no other use
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import gensim
model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

embeddings_dict = {}
with open("glove.twitter.27B.50d.txt", 'r',encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

df = pd.read_csv('ner_updated.csv', engine='python')
print(df['Tag'].value_counts())
df = df.groupby('Tag').apply(lambda x: x.sample(51))
#Encoding Labels
df['Tag'] = df['Tag'].map({'O':0,'B-art':1, 'I-art':2,'B-eve':3, 'I-eve':4,'B-geo':5, 'I-geo':6, 'B-gpe':7,
     'I-gpe':8, 'B-nat':9, 'I-nat':10, 'B-org':11, 'I-org':12, 'B-per':13, 'I-per':14, 'B-tim':15, 'I-tim':16})
np.random.seed(42)

settings = {
	'window_size': 2,
	'n': 300,		
	'epochs': 1,		
	'learning_rate': 0.001
}

embed = []
print(df)
X = df['Word']
y = df['Tag']
#Train Test Split
count = 0

for word,tag in zip(X,y):
    # if word in embeddings_dict:
    # x1 = embeddings_dict[word]
    try:
        count+=1
        x1 = model.wv[word]
        for x in x1:
            embed.append(x)
        embed.append(tag)
    except:
        continue


embeddf = pd.DataFrame(np.array(embed).reshape(-1, 301))
print(embeddf)
print(embeddf[50].value_counts())

X = embeddf.iloc[:,:settings['n']]

y = embeddf[settings['n']]

clf = DecisionTreeClassifier(random_state=0)
print('dt', cross_val_score(clf, X, y, cv=10))
neigh = KNeighborsClassifier(n_neighbors=5)
print('knn', cross_val_score(neigh, X, y, cv=10))
lr = LogisticRegression(random_state=0).fit(X, y)
print('lr', cross_val_score(lr, X, y, cv=10))




