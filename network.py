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
# print(df['Tag'].value_counts())
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
# print(df)
X = df['Word']
y = df['Tag']
#Train Test Split
count = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
for word,tag in zip(X_train,y_train):
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
# print(embeddf)
# print(embeddf[50].value_counts())

X = embeddf.iloc[:,:settings['n']]

y = embeddf[settings['n']]

one_hot_y = np.zeros((X.shape[0], 17))
i = 0
for j in y:
    one_hot_y[i, int(j)] = 1
    i+= 1


#Functions and Derivatives for the network
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(arr):
    expA = np.exp(arr)
    return expA / expA.sum(axis=1, keepdims=True)

#Specify parameters here, Vary the Hidden layer size
hidden_layer_size = 50
hidden_layer_size_2 = 100
no_of_classes = 17
#Initialize Weights and Biases
weights_hidden_layer = np.random.rand(X.shape[1],hidden_layer_size)
bias_hidden_layer = np.random.randn(hidden_layer_size)

weights_hidden_layer_2 = np.random.rand(hidden_layer_size,hidden_layer_size_2)
bias_hidden_layer_2 = np.random.randn(hidden_layer_size_2)

weights_output_layer = np.random.rand(hidden_layer_size_2,no_of_classes)
bias_output_layer = np.random.randn(no_of_classes)

print(weights_hidden_layer)

#learning rate for the classification network
cl_learn_rate = 0.001

error_cost = []

def predict(vec):
    # vec = we.wordvec(word) Get Vector
    vec = model.wv[word]
    z_hidden = np.dot(vec, weights_hidden_layer) + bias_hidden_layer
    act_hidden = sigmoid(np.dot(vec, weights_hidden_layer) + bias_hidden_layer)

    z_hidden_2 = np.dot(act_hidden, weights_hidden_layer_2) + bias_hidden_layer_2
    act_hidden_2 = sigmoid(z_hidden_2)

    z_output = np.dot(act_hidden_2, weights_output_layer) + bias_output_layer
    expA = np.exp(z_output)
    return expA / expA.sum()

for epoch in range(5000):
    #Feed Forward
    # Input to Hidden
    z_hidden = np.dot(X, weights_hidden_layer) + bias_hidden_layer
    act_hidden = sigmoid(z_hidden)

    # Hidden to Hidden2
    z_hidden_2 = np.dot(act_hidden, weights_hidden_layer_2) + bias_hidden_layer_2
    act_hidden_2 = sigmoid(z_hidden_2)
    # Hidden to Output
    z_output = np.dot(act_hidden_2, weights_output_layer) + bias_output_layer
    act_output = softmax(z_output)

    #BackProp
    #Output to Hidden
    del_output = act_output - one_hot_y
    del_weights_o = np.dot(act_hidden_2.T, del_output)

    #Hidden to Hidden
    del_w_o_2 = np.dot(del_output , weights_output_layer.T)
    der_hidden_2 = sigmoid_der(z_hidden_2)
    del_weights_h_2 = np.dot(act_hidden.T, der_hidden_2 * del_w_o_2)
    del_bias_h_2 = del_w_o_2 * der_hidden_2

    #Hidden to Input
    del_w_o = np.dot(del_w_o_2 , weights_hidden_layer_2.T)
    der_hidden = sigmoid_der(z_hidden)
    del_weights_h = np.dot(X.T, der_hidden * del_w_o)
    del_bias_h = del_w_o * der_hidden

    #Update Weights
    weights_hidden_layer -= cl_learn_rate * del_weights_h
    bias_hidden_layer -= cl_learn_rate * del_bias_h.sum(axis=0)

    weights_hidden_layer_2 -= cl_learn_rate * del_weights_h_2
    bias_hidden_layer_2 -= cl_learn_rate * del_bias_h_2.sum(axis=0)
    

    weights_output_layer -= cl_learn_rate * del_weights_o
    bias_output_layer -= cl_learn_rate * del_output.sum(axis=0)

    #Calculate Loss
    loss = np.sum(-one_hot_y * np.log(act_output))
    if epoch%200 == 0:
        print('Loss function value: ', loss)
    error_cost.append(loss)

count = 0
total = 0

ansdict = {0:[0,0,0,0], 1:[0,0,0,0], 2:[0,0,0,0], 3:[0,0,0,0], 4:[0,0,0,0], 5:[0,0,0,0], 6:[0,0,0,0], 7:[0,0,0,0], 8:[0,0,0,0],
    9:[0,0,0,0], 10:[0,0,0,0], 11:[0,0,0,0], 12:[0,0,0,0], 13:[0,0,0,0], 14:[0,0,0,0], 15:[0,0,0,0], 16:[0,0,0,0]}
for word,label in zip(X_test,y_test):
    # if word in embeddings_dict:
    try:
        model.wv[word]
        p = predict(word)
        ind = np.argmax(p)
        act_y = int(label)
        if ind == act_y:
            ansdict[act_y][0] += 1
            count +=1
        else:
            ansdict[ind][1] += 1
            ansdict[act_y][2] += 1
        total += 1
    except Exception as e:
        print(e)
        continue
        

for i in range(17):
    ansdict[i][3] = total - ansdict[i][0] - ansdict[i][1] - ansdict[i][2]
print('Confusion Matrix for each class', ansdict)
