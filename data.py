#Libraries permitted to Use
import numpy as np
import pandas as pd

#standard python libraries
import math
from collections import defaultdict

#This is just to perform the train test split on the dataset, no other use
from sklearn.model_selection import train_test_split



#This block preprocesses the dataset for the code ahead

# df = pd.read_csv('ner_dataset.csv', engine='python')
# print(df)
# last = 0
# n = 0
# print(type(df['Sentence #'][0]))
# for i in range(df.shape[0]):
#     print(i, last)
#     sen = str(df['Sentence #'][i])
#     if sen[0] != 'S':
#         df['Sentence #'][i] = int(last)
#     else:
#         s = str(df['Sentence #'][i])
#         last = str(s[9:])
#         df['Sentence #'][i] = int(last) 

# print(df)
# df.to_csv('ner_updated.csv')
df = pd.read_csv('ner_updated.csv', engine='python', nrows = 50000)
print(df)
# print(df['Tag'].value_counts())
# df = df.groupby('Tag').apply(lambda x: x.sample(51))
#Encoding Labels
df['Tag'] = df['Tag'].map({'O':0,'B-art':1, 'I-art':2,'B-eve':3, 'I-eve':4,'B-geo':5, 'I-geo':6, 'B-gpe':7,
     'I-gpe':8, 'B-nat':9, 'I-nat':10, 'B-org':11, 'I-org':12, 'B-per':13, 'I-per':14, 'B-tim':15, 'I-tim':16})


#Embedding Layer Settings
settings = {
	'window_size': 2,
	'n': 4,		
	'epochs': 5,		
	'learning_rate': 0.1
}
classification_epochs = 500


#Word Embedding Layer
class word_em():
    def __init__(self):
        self.ws = settings['window_size']
        self.n = settings['n']
        self.epochs = settings['epochs']
        self.lr = settings['learning_rate']
    
    #Generates the training data
    def gen_data(self, settings, df):
        word_count = defaultdict(int)
        for word in df['Word']:
            word_count[word] += 1
        self.v_count = len(word_count.keys())
        self.word_list = list(word_count.keys())
        self.word_index = dict((word, i) for i, word in enumerate(self.word_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.word_list))
        data = []
        for i in range(df['Sentence #'].max()):
            sent = df.loc[df['Sentence #'] == (i+1)]
            sent = list(sent['Word'])
            for i, word in enumerate(sent):
                w_t = self.onehot(sent[i])
                w_c = []
                for j in range(i-self.ws,i+self.ws):
                    if j <= len(sent)-1 and j != i and j >= 0:
                        w_c.append(self.onehot(sent[j]))
                data.append([w_t,w_c])
        return np.array(data)

    #One Hot Encoding for training
    def onehot(self, word):
        w_v = np.zeros(self.v_count)
        w_index = self.word_index[word]
        w_v[w_index] = 1
        return w_v
    
    #This trains the word embeddings
    def train(self, data):
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
        for i in range(self.epochs):
            self.loss = 0
            for w_t, w_c in data:
                y_pred, hidden, output = self.forward(w_t)
                error = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                self.backprop(error, hidden, w_t)
                self.loss += -np.sum([output[np.where(word == 1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(output)))
            print('Epoch:', i, "Loss:", self.loss)
    
    #Back-Propogation
    def backprop(self, error, hidden, w_v):
        dl_dw2 = np.outer(hidden, error)
        dl_dw1 = np.outer(w_v, np.dot(self.w2, error.T))
        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)

    #Feed Forward
    def forward(self, w_v):
        hidden = np.dot(self.w1.T, w_v)
        output = np.dot(self.w2.T, hidden)
        activated_output = self.softmax(output)
        return activated_output, hidden, output
    
    #Softmax
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    #This fetches the word vector
    def wordvec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

we = word_em()
data = we.gen_data(settings, df)
we.train(data)

#Test by getting a word vector
vec = we.wordvec("Thousands")
print(vec)

embed = []
# df = df.groupby('Tag').apply(lambda x: x.sample(9))
print(df)
X = df['Word']
y = df['Tag']
#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Extract word embeddingds
for word,tag in zip(X_train,y_train):
    x1 = we.wordvec(word)
    for x in x1:
        embed.append(x)
    embed.append(tag)

embeddf = pd.DataFrame(np.array(embed).reshape(-1, settings['n'] + 1))
print(embeddf)


#Neural Network for Classification, Two Hidden Layer
np.random.seed(42)
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

def relu(x):
    return max(x,0)

def relu_der(x):
    return 0 if x<=0 else 1

def softmax(arr):
    expA = np.exp(arr)
    return expA / expA.sum(axis=1, keepdims=True)

#Specify parameters here, Vary the Hidden layer size
hidden_layer_size = 50
hidden_layer_size_2 = 100
no_of_classes = 17
print(X.shape)
#Initialize Weights and Biases
weights_hidden_layer = np.random.rand(X.shape[1],hidden_layer_size)
bias_hidden_layer = np.random.randn(hidden_layer_size)

weights_output_layer = np.random.rand(hidden_layer_size, no_of_classes)
bias_output_layer = np.random.randn(no_of_classes)

#learning rate for the classification network
cl_learn_rate = 0.001

error_cost = []

def predict(vec):
    vec = we.wordvec(word) #Get Vector
    # vec = model.wv[word]
    z_hidden = np.dot(vec, weights_hidden_layer) + bias_hidden_layer
    act_hidden = sigmoid(np.dot(vec, weights_hidden_layer) + bias_hidden_layer)

    z_output = np.dot(act_hidden, weights_output_layer) + bias_output_layer
    expA = np.exp(z_output)
    return expA / expA.sum()

for epoch in range(classification_epochs):
    #Feed Forward
    # Input to Hidden
    z_hidden = np.dot(X, weights_hidden_layer) + bias_hidden_layer
    act_hidden = sigmoid(z_hidden)
    # Hidden to Output
    z_output = np.dot(act_hidden, weights_output_layer) + bias_output_layer
    act_output = softmax(z_output)

    #BackProp
    #Output to Hidden
    del_output = act_output - one_hot_y
    del_weights_o = np.dot(act_hidden.T, del_output)


    #Hidden to Input
    del_w_o = np.dot(del_output , weights_output_layer.T)
    der_hidden = sigmoid_der(z_hidden)
    del_weights_h = np.dot(X.T, der_hidden * del_w_o)
    del_bias_h = del_w_o * der_hidden

    #Update Weights
    weights_hidden_layer -= cl_learn_rate * del_weights_h
    bias_hidden_layer -= cl_learn_rate * del_bias_h.sum(axis=0)
    
    weights_output_layer -= cl_learn_rate * del_weights_o
    bias_output_layer -= cl_learn_rate * del_output.sum(axis=0)

    #Calculate Loss
    loss = np.sum(-one_hot_y * np.log(act_output))
    # if epoch%200 == 0:
        # print('Epoch', epoch, 'Loss function value: ', loss)
    error_cost.append(loss)

count = 0
total = 0

for word,label in zip(X_test,y_test):
    p = predict(word)
    ind = np.argmax(p)
    act_y = int(label)
    if ind == act_y:
        count +=1
    total += 1

print('Accuracy:', count/total)
