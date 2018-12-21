import sys
import os
import re
import numpy as np
import scipy
from scipy.spatial.distance import cosine
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import collections
from collections import Counter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

corpus = []
with open('train_pos.txt','r',encoding='latin1') as f:
  for line in f.readlines():
    corpus.append(line[:-1])

corpus = corpus[0:100]

length = len(corpus)

words = []
for sentence in corpus:
  words+=sentence.split()

vocab = list(set(words))

print(vocab)

vocab_len = len(vocab)
print(vocab_len)

word2idx = {}
index = 0
for word in vocab:
  word2idx[word] = index
  index+=1

print(word2idx)

idx2word = {}
index = 0
for word in vocab:
  idx2word[index] = word
  index+=1

print(idx2word)

window_size = 2
embedding_dim = 10

train_corpus = []
test_corpus = []

for sentence in corpus:
  words = sentence.split()
  sent_len = len(words)
  for i in range(sent_len):
    for j in range(max(i-window_size,0),min(i+window_size,sent_len)):
      train_corpus.append(word2idx[words[i]])
      test_corpus.append(word2idx[words[j]])

total_length = len(train_corpus)
print(total_length)

train = torch.zeros(total_length,vocab_len).to(device)
test = torch.zeros(total_length).to(device)

for i in range(total_length):
  train[i,train_corpus[i]] = 1
  test[i] = test_corpus[i]

class naive_softmax(nn.Module):
  def __init__(self):
    super(naive_softmax,self).__init__()
    self.vocabsize = vocab_len
    self.embeddingdim = embedding_dim
    self.linear1 = nn.Linear(self.vocabsize,self.embeddingdim)
    self.linear2 = nn.Linear(self.embeddingdim,self.vocabsize)
 
  def forward(self,x):
    out = self.linear1(x)
    out = self.linear2(out)
    out = F.log_softmax(out,dim=1)
    return out
    
  def predict(self,x):
    return self.linear1(x)

model = naive_softmax().to(device)

modeltest = torch.zeros((3,vocab_len)).to(device)
modeltest[0,0] = modeltest[1,1] = modeltest[2,2] = 1
output = model(modeltest)
print(output.shape)

traindata = torch.utils.data.TensorDataset(train,test)
trainloader = torch.utils.data.DataLoader(traindata,batch_size=64)

optimizer = optim.SGD(model.parameters(),lr=0.001)

numepochs = 20

model.train()
for epoch in range(numepochs):
  for i,(X,y) in enumerate(trainloader):
    X,y = X.to(device),y.to(device)
    optimizer.zero_grad()
    output = model(X)
    loss = F.nll_loss(output,y.long())
    loss.backward()
    optimizer.step()
    if(i%500==0):
      print("Epoch {} Batch {} Loss {}".format(epoch,i,loss))

with torch.no_grad():
  vocabvec = []
  vector = torch.zeros(vocab_len).to(device)
  for i in range(vocab_len):
      vector[i] = 1
      vocabvec.append(model.predict(vector).cpu().numpy())
      vector[i] = 0

def find_similarwords(targetword,numwords):
  with torch.no_grad():
    cosinedis = []
    vector = torch.zeros(vocab_len).to(device)
    vector[word2idx[targetword]] = 1
    targetvec = model.predict(vector)
    for i in range(vocab_len):
      cosinedis.append((cosine(targetvec,vocabvec[i]),i))
    sorted_dis = sorted(cosinedis,key=lambda x:x[0])
    simwords = []
    for i in range(numwords):
      simwords.append((sorted_dis[i][0],idx2word[i]))
    return simwords

similarwords = find_similarwords('great',10)

print(similarwords)

