import pymysql
import pandas as pd
import re
from datetime import datetime, timedelta
from collections import Counter
import os
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from gensim import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import DL_makemodel
import DL_makeclass
import sys

class config(object):
    embedding_dim = 300  # embedding vector size
    seq_length = 500  # maximum length of sequence
    vocab_size = 24000  # most common words

    num_filters = 200  # number of the convolution filters (feature maps)
    kernel_sizes = [3, 4, 5]  # three kind of kernels (windows)

    # hidden_dim = 128  # hidden size of fully connected layer

    dropout_prob = 0.5  # how much probability to be dropped
    learning_rate = 1e-3  # learning rate
    batch_size = 32  # batch size for training
    num_epochs = 20000  # total number of epochs
    hidden_size = 300
#    num_classes = len(list(set(list(data_train['FeedClass']))))   # number of classes
    num_classes = 5
    drop_prob = dropout_prob
    dev_split = 0.1  # percentage of dev data

#PATH="model0"
#ORDER='order.csv'
PATH = str(sys.argv[1])
ORDER = str(sys.argv[2])

mp='filtered'


label=['사회','연예','스포츠','경제','정치']
save_path='./'+PATH+'/'

print('read sample')
print(datetime.now())

data_train=pd.read_csv('./data/data_train_50.csv').drop_duplicates().sample(frac=1).reset_index(drop=True)
data_dev=pd.read_csv('./data/data_dev_50.csv').drop_duplicates().sample(frac=1).reset_index(drop=True)
data_test=pd.read_csv('./data/data_test_50.csv').drop_duplicates().sample(frac=1).reset_index(drop=True)



print('read fasttext')
print(datetime.now())

if PATH == 'model_facebook':
    embed_lookup = models.fasttext.load_facebook_model("./model/cc.ko.300.bin")
elif PATH == 'model_news':
    embed_lookup = models.fasttext.load_facebook_model("./model/news_text_model.bin")
else :
    embed_lookup = models.fasttext.load_facebook_model("./model/filtered_text_model.bin")
#embed_lookup = models.fasttext.load_facebook_model("./model/filtered__model.bin")

word2index={}
for i in range(0, len(embed_lookup.wv.index2word)):
    try:
        word2index[embed_lookup.wv.index2word[i]] = i
    except KeyError:
        word2index[embed_lookup.wv.index2word[i]] = i

weights = list()
for i in range(0, len(embed_lookup.wv.vocab)):
    cc = embed_lookup.wv.index2word[i]
    if(i == 0):
        weights.append(np.ndarray.tolist(np.zeros(300,)))
        continue
    try:
        weights.append(np.ndarray.tolist(embed_lookup[cc]))
    except KeyError:
        weights.append(np.ndarray.tolist(np.zeros(300,)))   
    
weights = np.array(weights, dtype=np.float32)
weights = torch.from_numpy(weights)


y_train = list(data_train['FeedClass'])
y_dev = list(data_dev['FeedClass'])
y_test = list(data_test['FeedClass'])
    
x_train = list(data_train[mp])
x_dev = list(data_dev[mp])
x_test = list(data_test[mp])

#order_list=pd.read_csv('/data/public/splunk_DL/taitans/0701/test1/'+ORDER) 
#for i in [0,1]:
order_list=pd.read_csv('./'+ORDER) 
for i in range(0,len(order_list)):

    text_train=[]
    text_train_len=[]
    text_test=[]
    text_test_len=[]
    text_dev=[]
    text_dev_len=[]

#    config.embedding_dim=order_list['embedding_dim'][i]
#    config.seq_length=order_list['seq_length'][i]
#    config.num_filters=order_list['num_filters'][i]
#    config.dropout_prob=order_list['dropout_prob'][i]
#    config.learning_rate=order_list['learning_rate'][i]
#    config.hidden_size=order_list['hidden_size'][i]
    es=order_list['EarlyStop'][i]
    running=order_list['model'][i]
#    es=3
#    RRR=["Bi_A_Lstm","Bi_A_Gru"]
#    running=RRR[i]

    config.num_classes=len(list(set(y_train)))

    for i in range(0,len(x_train)):  # tokenizing and padding
        train_text, train_len = DL_makemodel.process_text(x_train[i], word2index, config.seq_length)
        text_train.append(train_text)
        text_train_len.append(train_len)

    text_train = np.array(text_train)

    for i in range(0,len(x_dev)):  # tokenizing and padding
        dev_text, dev_len = DL_makemodel.process_text(x_dev[i], word2index, config.seq_length)
        text_dev.append(dev_text)
        text_dev_len.append(dev_len)
    text_dev = np.array(text_dev)

    for i in range(0,len(x_test)):  # tokenizing and padding
        test_text, test_len = DL_makemodel.process_text(x_test[i], word2index, config.seq_length)
        text_test.append(test_text)
        text_test_len.append(test_len)
    text_test = np.array(text_test)
    #print(datetime.now())
    label_train=[]
    label_test=[]
    label_dev=[]

    for i in range(0,len(y_train)):
        label_train.append(label.index(y_train[i]) if y_train[i] in label else "notinlist")
    for i in range(0,len(y_test)):
        label_test.append(label.index(y_test[i]) if y_test[i] in label else "notinlist")
    for i in range(0,len(y_dev)):
        label_dev.append(label.index(y_dev[i]) if y_dev[i] in label else "notinlist")

    label_train = np.array(label_train)
    label_test = np.array(label_test)
    label_dev = np.array(label_dev)

    TD_train=TensorDataset(torch.LongTensor(text_train), torch.LongTensor(label_train), torch.LongTensor(text_train_len))
    TD_test=TensorDataset(torch.LongTensor(text_test), torch.LongTensor(label_test), torch.LongTensor(text_test_len))
    TD_dev=TensorDataset(torch.LongTensor(text_dev), torch.LongTensor(label_dev), torch.LongTensor(text_dev_len))
    #TD_real=TensorDataset(torch.LongTensor(text_dev), torch.LongTensor(label_dev), torch.LongTensor(text_dev_len))
    #print(datetime.now())
    st=datetime.now()
    model,acc,DL=DL_makemodel.train(running,TD_train,TD_dev, TD_test,weights,es,save_path, config)
    df=pd.DataFrame([str(datetime.now()),config.embedding_dim,config.seq_length,config.num_filters,config.dropout_prob,config.learning_rate,config.hidden_size,es,running,acc,DL,str(datetime.now()-st)]).T
    df.columns=['time','embedding_dim','seq_length','num_filters','dropout_prob','learning_rate','hidden_size','EarlyStop','model','acc','dev_loss','running_time']
    #result=pd.read_csv('./result.csv')
#    Final_df=pd.concat([result,df])
    Final_df=df
    Final_df.to_csv('./result.csv',index=False)
    print(Final_df)
