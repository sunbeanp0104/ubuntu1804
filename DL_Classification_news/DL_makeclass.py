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


class TextGRU2(nn.Module):

    def __init__(self, config):
        super(TextGRU2, self).__init__()
        V = config.vocab_size
        E = config.embedding_dim
        H = config.hidden_size
        D = config.drop_prob
        C = config.num_classes
        self.n_hidden = config.hidden_size

        self.gru = nn.GRU(E, H, batch_first=True)
        self.drop = nn.Dropout(D)
        self.classifier = nn.Linear(H, C)


    def attention_net(self,lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state,soft_attn_weights

    def forward(self, x,x_batch_len):
        packed_input = pack_padded_sequence(x, x_batch_len.cpu().numpy(), batch_first=True)
        packed_output,final_hidden_state= self.gru(packed_input)
        
        r_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        #attn_output,soft_attn_weights = self.attention_net(r_output, r_output[:, -1, :])
        attn_output,soft_attn_weights = self.attention_net(r_output,final_hidden_state)
        logits = self.classifier(attn_output)
        return F.log_softmax(logits)

# class Bi_Attention_GRU(nn.Module):
#   def __init__(self, config):
#   super(Bi_Attention_GRU, self).__init__()
# V = config.vocab_size
# E = config.embedding_dim
# H = config.hidden_size
# D = config.drop_prob
# C = config.num_classes
# self.n_hidden = config.hidden_size
# self.H = H
# 
# self.gru = nn.GRU(E, H, batch_first=True, bidirectional= True)
# self.drop = nn.Dropout(D)
# 
# self.classifier = nn.Linear(2*H, C)
# 
# def attention_net(self,lstm_output, final_state):
#   hidden = final_state.squeeze(0)
# attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
# soft_attn_weights = F.softmax(attn_weights, 1)
# new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
# return new_hidden_state,soft_attn_weights
# 
# def forward(self, x,x_batch_len):
#   packed_input = pack_padded_sequence(x, x_batch_len.cpu().numpy(), batch_first=True)
# packed_output,final_hidden_state= self.gru(packed_input)
# r_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
# final_hidden_state = torch.cat((final_hidden_state[0],final_hidden_state[1]),1)   
# 
# #attn_output,soft_attn_weights = self.attention_net(r_output, r_output[:, -1, :])
# attn_output,soft_attn_weights = self.attention_net(r_output,final_hidden_state)
# logits = self.classifier(attn_output)
# return F.log_softmax(logits)

class Bi_A_Gru(nn.Module):

    def __init__(self, config):
        super(Bi_A_Gru, self).__init__()
        V = config.vocab_size
        E = config.embedding_dim
        H = config.hidden_size
        D = config.drop_prob
        C = config.num_classes
        self.n_hidden = config.hidden_size
        self.H = H

        self.gru = nn.GRU(E, H, batch_first=True, bidirectional= True)
        # self.dropout = nn.Dropout(D)
        self.drop = nn.Dropout(0.5)
        
        self.classifier = nn.Linear(2*H, C)
        

        self.sig = nn.Sigmoid()
        self.Lin1 = nn.Linear(2 * H, C)

    def attention_net(self,lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state,soft_attn_weights

    def forward(self, x,x_batch_len):
        packed_input = pack_padded_sequence(x, x_batch_len.cpu().numpy(), batch_first=True)
        packed_output,final_hidden_state= self.gru(packed_input)
        r_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        final_hidden_state = torch.cat((final_hidden_state[0],final_hidden_state[1]),1)   
        
        #attn_output,soft_attn_weights = self.attention_net(r_output, r_output[:, -1, :])
        attn_output,soft_attn_weights = self.attention_net(r_output,final_hidden_state)
        logits = self.classifier(attn_output)
        return F.log_softmax(logits)
    
class Bi_A_Rnn(nn.Module):

    def __init__(self, config):
        super(Bi_A_Rnn, self).__init__()
        V = config.vocab_size
        E = config.embedding_dim
        H = config.hidden_size
        D = config.drop_prob
        C = config.num_classes
        self.n_hidden = config.hidden_size
        self.H = H

        self.gru = nn.RNN(E, H, batch_first=True, bidirectional= True)
        # self.dropout = nn.Dropout(D)
        self.drop = nn.Dropout(0.5)
        
        self.classifier = nn.Linear(2*H, C)
        

        self.sig = nn.Sigmoid()
        self.Lin1 = nn.Linear(2 * H, C)

    def attention_net(self,lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state,soft_attn_weights

    def forward(self, x,x_batch_len):
        packed_input = pack_padded_sequence(x, x_batch_len.cpu().numpy(), batch_first=True)
        packed_output,final_hidden_state= self.gru(packed_input)
        r_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        final_hidden_state = torch.cat((final_hidden_state[0],final_hidden_state[1]),1)   
        
        #attn_output,soft_attn_weights = self.attention_net(r_output, r_output[:, -1, :])
        attn_output,soft_attn_weights = self.attention_net(r_output,final_hidden_state)
        logits = self.classifier(attn_output)
        return F.log_softmax(logits)


class TextLSTM2(nn.Module):

    def __init__(self, config):
        super(TextLSTM2, self).__init__()
        V = config.vocab_size
        E = config.embedding_dim
        H = config.hidden_size
        D = config.drop_prob
        C = config.num_classes
        self.n_hidden = config.hidden_size

        self.lstm = nn.LSTM(E, H, batch_first=True)
        self.drop = nn.Dropout(D)
        self.classifier = nn.Linear(H, C)

    def attention_net(self,lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state,soft_attn_weights

    def forward(self, x,x_batch_len):
        #print("inputs##2222:", x_batch_len)
        packed_input = pack_padded_sequence(x, x_batch_len.cpu().numpy(), batch_first=True)
        packed_output,(final_hidden_state, final_cell_state) = self.lstm(packed_input)
        
        r_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        attn_output,soft_attn_weights = self.attention_net(r_output, final_hidden_state)
        logits = self.classifier(attn_output)
        return F.log_softmax(logits)



class Bi_A_Lstm(nn.Module):
    def __init__(self, config):
        super(Bi_A_Lstm, self).__init__()
        V = config.vocab_size
        E = config.embedding_dim
        H = config.hidden_size
        D = config.drop_prob
        C = config.num_classes
        self.n_hidden = config.hidden_size
        self.H = H

        self.lstm = nn.LSTM(E, H,batch_first=True, bidirectional= True)
        # self.dropout = nn.Dropout(D)
        self.drop = nn.Dropout(0.5)
        
        self.classifier = nn.Linear(2*H, C)
        

        self.sig = nn.Sigmoid()
        self.Lin1 = nn.Linear(2 * H, C)

    def attention_net(self,lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(1)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights).squeeze(2)
#        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state,soft_attn_weights

    def forward(self, x,x_batch_len):
        packed_input = pack_padded_sequence(x, x_batch_len.cpu().numpy(), batch_first=True)
        packed_output,(final_hidden_state, final_cell_state)= self.lstm(packed_input)
        r_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        final_hidden_state = torch.cat((final_hidden_state[0],final_hidden_state[1]),1)   
        attn_output,soft_attn_weights = self.attention_net(r_output,final_hidden_state)
        logits = self.classifier(attn_output)
        return F.log_softmax(logits)
    
    

class TextRNN2(nn.Module):

    def __init__(self, config):
        super(TextRNN2, self).__init__()
        V = config.vocab_size
        E = config.embedding_dim
        H = config.hidden_size
        D = config.drop_prob
        C = config.num_classes
        self.n_hidden = config.hidden_size
        self.lstm = nn.RNN(E, H, batch_first=True)
        self.drop = nn.Dropout(D)
        self.classifier = nn.Linear(H, C)

    def attention_net(self,lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state,soft_attn_weights

    def forward(self, x,x_batch_len):
        packed_input = pack_padded_sequence(x, x_batch_len.cpu().numpy(), batch_first=True)
        packed_output,final_hidden_state = self.lstm(packed_input)
        r_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        attn_output,soft_attn_weights = self.attention_net(r_output, final_hidden_state)
        logits = self.classifier(attn_output)
        return F.log_softmax(logits)


    
class TextLSTM(nn.Module):

    def __init__(self, config):
        super(TextLSTM, self).__init__()
        V = config.vocab_size
        E = config.embedding_dim
        H = config.hidden_size
        D = config.drop_prob
        C = config.num_classes
        self.lstm = nn.LSTM(E, H, num_layers=1, batch_first=True,bidirectional= True)
#        self.lstm = nn.LSTM(E, H, num_layers=1, batch_first=True)
#        self.sig = nn.Sigmoid()
        self.Lin1 = nn.Linear(H*2, C)
#        self.Lin1 = nn.Linear(H, C)

    def forward(self, x,x_batch_len):
        packed_input = pack_padded_sequence(x, x_batch_len.cpu().numpy(), batch_first=True)
        packed_output,(final_hidden_state, final_cell_state) = self.lstm(packed_input)
        r_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        r_output = torch.transpose(r_output, 2,1)
        output = F.tanh(r_output)
        final_output = F.max_pool1d(output, output.size(2)).squeeze(2)
        final_output = self.Lin1(final_output)
        return F.log_softmax(final_output)


    
class TextRNN(nn.Module):

    def __init__(self, config):
        super(TextRNN, self).__init__()
        V = config.vocab_size
        E = config.embedding_dim
        H = config.hidden_size
        D = config.drop_prob
        C = config.num_classes
        self.hidden = config.hidden_size                  #?
        self.dropout = nn.Dropout(D)                      #?
        self.rnn = nn.RNN(E, H, num_layers=1, batch_first=True,bidirectional= True)
        self.sig = nn.Sigmoid()                           # notuse
        self.Lin1 = nn.Linear(H*2, C)

    def forward(self, x,x_batch_len):
        packed_input = pack_padded_sequence(x, x_batch_len.cpu().numpy(), batch_first=True)
        packed_output,(final_hidden_state, final_cell_state) = self.rnn(packed_input)
        r_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        r_output = torch.transpose(r_output, 2,1)
        output = F.tanh(r_output)
        final_output = F.max_pool1d(output, output.size(2)).squeeze(2)
        final_output = self.Lin1(final_output)
        return F.log_softmax(final_output)

'''
    
class TextRNN(nn.Module):

    def __init__(self, config):
        super(TextRNN, self).__init__()
        V = config.vocab_size
        E = config.embedding_dim
        H = config.hidden_size
        D = config.drop_prob
        C = config.num_classes
        self.hidden = config.hidden_size

        self.rnn = nn.RNN(E, H, num_layers=1, batch_first=True,bidirectional= True)
#        self.rnn = nn.RNN(E, H, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(D)

        self.sig = nn.Sigmoid()
        self.Lin1 = nn.Linear(H*2, C)
#        self.Lin1 = nn.Linear(H, C)

    def forward(self, x,x_batch_len):
        packed_input = pack_padded_sequence(x, x_batch_len.cpu().numpy(), batch_first=True)
        packed_output,final_hidden_state = self.rnn(packed_input)
        r_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        r_output = torch.transpose(r_output, 2,1)
        output = F.tanh(r_output)
        final_output = F.max_pool1d(output, output.size(2)).squeeze(2)
        
        return F.log_softmax(final_output)
'''
class TextGRU(nn.Module):
    def __init__(self, config):
        super(TextGRU, self).__init__()
        V = config.vocab_size
        E = config.embedding_dim
        H = config.hidden_size
        D = config.drop_prob
        C = config.num_classes
        self.hidden = config.hidden_size                  #?
        self.dropout = nn.Dropout(D)                      #?
        self.gru = nn.GRU(E, H, num_layers=1, batch_first=True,bidirectional= True)
        self.sig = nn.Sigmoid()                           # notuse
        self.Lin1 = nn.Linear(H*2, C)

    def forward(self, x,x_batch_len):
        packed_input = pack_padded_sequence(x, x_batch_len.cpu().numpy(), batch_first=True)
        packed_output,(final_hidden_state, final_cell_state) = self.gru(packed_input)
        r_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        r_output = torch.transpose(r_output, 2,1)
        output = F.tanh(r_output)
        final_output = F.max_pool1d(output, output.size(2)).squeeze(2)
        final_output = self.Lin1(final_output)
        return F.log_softmax(final_output)
'''
class TextGRU(nn.Module):

    def __init__(self, config):
        super(TextGRU, self).__init__()
        V = config.vocab_size
        E = config.embedding_dim
        H = config.hidden_size
        D = config.drop_prob
        C = config.num_classes
#        self.gru = nn.GRU(E, H, num_layers=1, batch_first=True,bidirectional= True)
        self.gru = nn.GRU(E, H, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(D)

        self.sig = nn.Sigmoid()
#        self.Lin1 = nn.Linear(H*2, C)
        self.Lin1 = nn.Linear(H, C)

    def forward(self, x,x_batch_len):
        packed_input = pack_padded_sequence(x, x_batch_len.cpu().numpy(), batch_first=True)
        packed_output,final_hidden_state = self.gru(packed_input)
        r_output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        r_output = torch.transpose(r_output, 2,1)
        output = F.tanh(r_output)
        final_output = F.max_pool1d(output, output.size(2)).squeeze(2)
        return F.log_softmax(final_output)
'''  
                
class TextCNN(nn.Module):

    def __init__(self, config):
        super(TextCNN, self).__init__()

        V = config.vocab_size
        E = config.embedding_dim
        Nf = config.num_filters
        Ks = config.kernel_sizes
        C = config.num_classes
        Dr = config.dropout_prob

        self.embedding = nn.Embedding(V, E)  # embedding layer

        # three different convolutional layers
        # 이것은 간단히 말해 nn.Module을 리스트로 정리하는 방법이다. 각 레이어를 리스트에 전달하고 레이어의 iterator를 만든다. 덕분에 forward처리를 간단하게 할 수 있다는 듯 하다. 처음으로 적는 것은 아주 무식하게 하나하나 다 적어서 리스트에 넣고 for로 돌리는 방식이다.
        self.convs = nn.ModuleList([nn.Conv1d(E, Nf, k) for k in Ks])
        self.dropout = nn.Dropout(Dr)  # a dropout layer
        self.fc1 = nn.Linear(3 * Nf, C)  # a dense layer for classification

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs):
        # Conv1d takes in (batch, channels, seq_len), but raw embedded is (batch, seq_len, channels)
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x = [self.conv_and_max_pool(embedded, k) for k in self.convs]  # convolution and global max pooling #
        x = self.fc1(self.dropout(torch.cat(x, 1)))  # concatenation and dropout 300 -> 2차원으로 축소

        return x
    
