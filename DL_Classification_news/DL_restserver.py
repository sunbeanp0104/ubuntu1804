#-*- codeing:utf-8 -*-
from time import sleep
import logging.handlers
from flask import Flask as f, request
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import TimedRotatingFileHandler

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
from tqdm import tqdm
from gensim import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import DL_makeclass
use_cuda = torch.cuda.is_available()

run="Lstm"
text=['서울,연합뉴스,권수현,기자,연휴,이후,코스피,연일,사상,최고,치를,경신,하는,국내,증시,강세,어가,주가,하락,염두,공매도,투자자,적잖,낭패,으로,나타났,23일,한국,거래소,코스콤,따르,추석,연휴,이후,10,20일,코스피,코스닥,시장,에서,누적,거래,대비,공매도,거래,비중,종목,상위,20,가운데,연휴']

print('read fasttext')
#embed_lookup = models.fasttext.load_facebook_model("/data/public/model/cc.ko.300.bin")
embed_lookup = models.fasttext.load_facebook_model("/data/public/model/news_text_model.bin")

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

print("#weights :", len(weights))

def process_text(text, word_to_id, max_length):
    text = text.split(",")
    #print(len(text))
    text2 = []
    for word in text:
        try:
            idx = word_to_id[word]
            #idx = token2idx[word]
        except: 
            idx = 0
        text2.append(idx)

            

    #text = [word_to_id[x] for x in text if x in word_to_id]
    len_text = len(text2)
    if len(text2) < max_length:
        text2 = text2 +  [0] * (max_length - len(text2)) 
    else:
        len_text = max_length

    return text2[:max_length], len_text



def predictfunction(run,text):

    FeedClass=['사회','연예','스포츠','경제','정치']

    class config(object):

        embedding_dim = 300  # embedding vector size
        seq_length = 500  # maximum length of sequence
        vocab_size = 24000  # most common words

        num_filters = 200  # number of the convolution filters (feature maps)
        kernel_sizes = [3, 4, 5]  # three kind of kernels (windows)

        # hidden_dim = 128  # hidden size of fully connected layer

        dropout_prob = 0.5  # how much probability to be dropped
        learning_rate = 1e-7  # learning rate
        batch_size = 32  # batch size for training
        num_epochs = 20000  # total number of epochs
        hidden_size = 300
    #    num_classes = len(list(set(list(data_train['FeedClass']))))   # number of classes
        num_classes = 5
        drop_prob = 0.6
        dev_split = 0.1  # percentage of dev data


    if run == 'A_Gru':
        model = DL_makeclass.TextGRU2(config)
    elif run == 'A_Lstm':
        model = DL_makeclass.TextLSTM2(config)
    elif run == 'A_Rnn':
        model = DL_makeclass.TextRNN2(config)
    elif run == 'Lstm':
        model = DL_makeclass.TextLSTM(config)
    elif run == 'Rnn':
        model = DL_makeclass.TextRNN(config)
    elif run == 'Gru':
        model = DL_makeclass.TextGRU(config)
    elif run == 'Cnn':
        model = DL_makeclass.TextCNN(config)
    elif run == 'Bi_A_Rnn':
        model = DL_makeclass.Bi_A_Rnn(config)
    elif run == 'Bi_A_Gru':
        model = DL_makeclass.Bi_A_Gru(config)
    elif run == 'Bi_A_Lstm':
        model = DL_makeclass.Bi_A_Lstm(config)
    model.cuda()
    model.load_state_dict(torch.load("/data/public/model/DL_"+run+".pt"))
    print(model)
    #def predict_test(model,text , run,config):
    text_test=[]
    text_test_len=[]

#    for i in range(len(text)):  # tokenizing and padding
#        test_text, test_len = process_text(text[i], word2index, config.seq_length)
    test_text, test_len = process_text(text, word2index, config.seq_length)
    text_test.append(test_text)
    text_test_len.append(test_len)

    text_test = np.array(text_test)

    #print(datetime.now())
    label_test = np.array([0])

    TD_test=TensorDataset(torch.LongTensor(text_test), torch.LongTensor(label_test), torch.LongTensor(text_test_len))




    print("predict....")
    start_time = time.time()
    test_loader = DataLoader(TD_test, batch_size=50)
    # restore the best parameters
    if use_cuda:
        print("############################################## use cuda ###########")
        model.cuda()
    y_true, y_pred = [], []
    y_true1 = []
    data_len = len(test_loader)

    for data, label, inputs_len  in test_loader:

        embedding = nn.Embedding.from_pretrained(weights)
        seq_lengths, perm_idx = inputs_len.sort(0, descending=True)
        data = data[perm_idx]
        label = label[perm_idx]
        data = embedding(data.long())

        if use_cuda:
            data, label, seq_lengths = data.cuda(), label.cuda(), seq_lengths.cuda()


        output = model(data, seq_lengths)
        #print(output)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)

    return list(map(lambda x: FeedClass[x],y_pred))[0]
app = f(__name__)

@app.route('/help',methods=['GET', 'POST'])
def printtext():
    printlist=['http://112.220.110.133:12345/predict?run=A_Lstm&text=서울,연합뉴스,권수현,기자,연휴,이후,코스피,연일,사상,최고,치를,경신,하는,국내'],
    return(print(printlist))

@app.route('/predict', methods=['GET', 'POST'])
#def predictfunction(run,text,word2index,weights):
def PreDict():
    content = request.args
    print(content)
    run=content['run']
    print(run)
    text=content['text']

    return predictfunction(run,text)


if __name__ == "__main__":              
    app.run(host="112.220.110.133",port="12345")
