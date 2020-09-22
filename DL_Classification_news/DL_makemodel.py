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
from tqdm import tqdm
from gensim import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import DL_makeclass
use_cuda = torch.cuda.is_available()

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0,run='test',sp='modelsave'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = os.path.join(sp, 'DL_' + run + '.pt')       

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter:', self.counter, 'out of', self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            torch.save(model.state_dict(), self.path)


    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print('Validation loss decreased ',self.val_loss_min,val_loss,  'Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def evaluate(data, model, loss,weights,config):
    
    model.eval()  # set mode to evaluation to disable dropout
    data_loader = DataLoader(data, batch_size=config.batch_size)

    data_len = len(data)
    val_loss = []
    y_true, y_pred = [], []

    for data, label, inputs_len  in data_loader:
        embedding = nn.Embedding.from_pretrained(weights)
        
        seq_lengths, perm_idx = inputs_len.sort(0, descending=True)
        data = data[perm_idx]
        label = label[perm_idx]
        data = embedding(data.long())
        
        if use_cuda:
            data, label, inputs_len = data.cuda(), label.cuda(), inputs_len.cuda()
            
        output = model(data, seq_lengths)
        losses = loss(output, label)

        val_loss.append(losses.item())

        y_pred2 = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(y_pred2)
        y_true.extend(label.data)

    
        
    acc = (np.array(y_true) == np.array(y_pred)).sum()

    return acc / data_len, np.mean(val_loss)

def testA(model, test_dataset, run,weights):

    print("Testing...")
    start_time = time.time()
    test_loader = DataLoader(test_dataset, batch_size=50)
    # restore the best parameters
    if use_cuda:
        print("==============================================================================================")
        print("=============== use cuda ================= use cuda ================= use cuda================")
        print("==============================================================================================")
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
        y_true.extend(label.data)
  
    for i in range(0, len(y_true)):
        y_true1.append(y_true[i].item())
    
    acc = (np.array(y_true) == np.array(y_pred)).sum()
    
    test_acc = accuracy_score(y_true1, y_pred)

    test_f1 = metrics.f1_score(y_true1, y_pred, average='macro')

    print("Test accuracy: {0:>7.2%}, F1-score: {0:>7.2%}".format(test_acc).format(test_f1))
    
    return test_acc





    
def train(run,train_data,dev_data, test_data,weights,es,save_path,config):
    # def train():
    print("test_data#:", test_data)
    model_file = os.path.join(save_path, 'DL_' + run + '.pt')

    """
    Train and evaluate the model with training and validation data.
    """
    print('Loading data...')
    start_time = time.time()

    # print('Configuring CNN model...')
    print('Configuring ' + run + ' model...')
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
    elif run == 'Bi_A_Gru':
        model = DL_makeclass.Bi_A_Gru(config)
    elif run == 'Bi_A_Lstm':
        model = DL_makeclass.Bi_A_Lstm(config)
    elif run == 'Bi_A_Rnn':
        model = DL_makeclass.Bi_A_Rnn(config)
        
        

    print(model)

    if use_cuda:
        print("==============================================================================================")
        print("=============== use cuda ================= use cuda ================= use cuda================")
        print("==============================================================================================")
        model.cuda()

    # optimizer and loss function
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # set the mode to train
    print("Training and evaluating...")
    
    best_acc = 0.0
    acc_update = 0
    early_stopping = EarlyStopping(patience=es, verbose=True,run=run,sp=save_path)
    for epoch in range(config.num_epochs):
        # load the training data in batch
        loss_epoch = []
        
        model.train()
        train_loader = DataLoader(train_data, batch_size=config.batch_size)
        for x_batch, y_batch, x_batch_len in train_loader:
            #print(x_batch)
            embedding = nn.Embedding.from_pretrained(weights)
            seq_lengths, perm_idx = x_batch_len.sort(0, descending=True)
            inputs = x_batch[perm_idx]
            
            targets = y_batch[perm_idx]
            inputs = embedding(inputs.long())
           
            if use_cuda:
                inputs, targets, inputs_len = inputs.cuda(), targets.cuda(), x_batch_len.cuda()
            
            
            optimizer.zero_grad()

            outputs = model(inputs,seq_lengths)  # forward computation
           # print(outputs[0])
            loss = criterion(outputs, targets)

            
  
            loss_epoch.append(loss.item())

            # backward propagation and update parameters
            loss.backward()
            optimizer.step()
            
        print ('Average training loss at this epoch..minibatch ' ,  np.mean(loss_epoch))
        
        # evaluate on both training and test dataset
        
        train_acc, train_loss = evaluate(train_data, model, criterion,weights,config)
        dev_acc, dev_loss = evaluate(dev_data, model, criterion,weights,config)
        
        print("epoch: ", epoch, " train_loss: ", train_loss, "train_acc: ", train_acc, "dev_loss: ", dev_loss, "dev_acc: ", dev_acc,"time :",datetime.now())
       
        
        #test_acc, test_loss = evaluate(test_data, model, criterion)
        #testA(model,test_data)
        early_stopping(dev_loss, model)
        torch.cuda.empty_cache()
        

        if early_stopping.early_stop:
            print("Early stopping")
            model.eval()
            model.load_state_dict(torch.load(save_path+'DL_' + run + '.pt'))
            acc=testA(model, test_data, run,weights)
            return model, acc, dev_loss
            break
            
            
            
def process_text(text, word2index, max_length):
    text = text.split(",")
    #print(len(text))
    text2 = []
    for word in text:
        try:
            idx = word2index[word]
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
