#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.2.24

'''
Seq2Seq with Attention + GP prior
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import math
import random
import numpy as np
import pickle
import ipdb
import sys

from .layers import * 


class Encoder(nn.Module):
    
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5, pretrained=None):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.embed_size = embed_size
        self.n_layer = n_layers
        
        self.embed = nn.Embedding(self.input_size, self.embed_size)
        # self.input_dropout = nn.Dropout(p=dropout)
        
        self.rnn = nn.GRU(embed_size, 
                          hidden_size, 
                          num_layers=n_layers, 
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True)

        # self.hidden_proj = nn.Linear(2 * n_layers * hidden_size, hidden_size)
        # self.bn = nn.BatchNorm1d(num_features=hidden_size)
            
        self.init_weight()
            
    def init_weight(self):
        # orthogonal init
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)
        
    def forward(self, src, inpt_lengths, hidden=None):
        # src: [seq, batch]
        embedded = self.embed(src)    # [seq, batch, embed]
        # embedded = self.input_dropout(embedded)

        if not hidden:
            hidden = torch.randn(2 * self.n_layer, src.shape[-1], self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()
        
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, inpt_lengths, 
                                                     enforce_sorted=False)
        # hidden: [2 * n_layer, batch, hidden]
        # output: [seq_len, batch, 2 * hidden_size]
        output, hidden = self.rnn(embedded, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # fix output shape
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        # hidden = hidden.sum(axis=0)    # [batch, hidden]
        
        # fix hidden
        # hidden = hidden.permute(1, 0, 2)
        # hidden = hidden.reshape(hidden.shape[0], -1)
        # hidden = self.bn(hidden)    # [batch, *]
        # hidden = self.hidden_proj(hidden)
        hidden = torch.tanh(hidden)
        
        # [seq_len, batch, hidden_size], [batch, hidden]
        return output, hidden
    
    
class Decoder(nn.Module):
    
    def __init__(self, embed_size, hidden_size, output_size, n_layers=2, dropout=0.5, pretrained=None):
        super(Decoder, self).__init__()
        self.embed_size, self.hidden_size = embed_size, hidden_size
        self.output_size = output_size
        
        self.embed = nn.Embedding(output_size, embed_size)
        self.attention = Attention(hidden_size) 
        self.rnn = nn.GRU(hidden_size + embed_size, 
                          hidden_size,
                          num_layers=n_layers, 
                          dropout=(0 if n_layers == 1 else dropout))
        self.out = nn.Linear(hidden_size, output_size)
        
        self.init_weight()
        
    def init_weight(self):
        # orthogonal inittor
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)
        
    def forward(self, inpt, last_hidden, encoder_outputs):
        # inpt: [batch]
        # last_hidden: [2, batch, hidden_size]
        embedded = self.embed(inpt).unsqueeze(0)    # [1, batch, embed_size]
        
        # attn_weights: [batch, 1, timestep of encoder_outputs]
        key = last_hidden.sum(axis=0)
        attn_weights = self.attention(key, encoder_outputs)
            
        # context: [batch, 1, hidden_size]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)
        
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(0)
        # context = context.squeeze(0)
        # [batch, hidden * 2]
        # output = self.out(torch.cat([output, context], 1))
        output = self.out(output)    # [batch, output_size]
        output = F.log_softmax(output, dim=1)
        
        # output: [batch, output_size]
        # hidden: [2, batch, hidden_size]
        # hidden = hidden.squeeze(0)
        return output, hidden
    
    
class Seq2Seq_Full(nn.Module):
    
    '''
    Compose the Encoder and Decoder into the Seq2Seq model
    '''
    
    def __init__(self, input_size, embed_size, output_size, 
                 utter_hidden, decoder_hidden, latent_size, v, r, 
                 teach_force=0.5, pad=24745, sos=24742, dropout=0.5, 
                 utter_n_layer=1, src_vocab=None, tgt_vocab=None,
                 pretrained=None):
        super(Seq2Seq_Full, self).__init__()
        self.encoder = Encoder(input_size, embed_size, utter_hidden,
                               n_layers=utter_n_layer, 
                               dropout=dropout,
                               pretrained=pretrained)
        self.decoder = Decoder(embed_size, decoder_hidden, 
                               output_size, n_layers=utter_n_layer,
                               dropout=dropout,
                               pretrained=pretrained)
        self.teach_force = teach_force
        self.utter_n_layer = utter_n_layer
        self.pad, self.sos = pad, sos
        self.output_size = output_size
        
        self.latent_size = latent_size
        self.using_cuda = torch.cuda.is_available()
        self.mean = nn.Linear(self.encoder.hidden_size, self.latent_size)
        self.logvar = nn.Linear(self.encoder.hidden_size, self.latent_size)
        self.latent2hidden = nn.Linear(self.latent_size, self.decoder.hidden_size)
        
        # parameters for kernel
        kernel_v = nn.Parameter(torch.ones(1)* v, requires_grad=False)
        self.register_parameter('kernel_v', kernel_v)
        kernel_r = nn.Parameter(torch.ones(1)* r, requires_grad=False)
        self.register_parameter('kernel_r', kernel_r)
        
    def forward(self, src, tgt, lengths):
        # src: [lengths, batch], tgt: [lengths, batch], lengths: [batch]
        # ipdb.set_trace()
        batch_size, max_len = src.shape[1], tgt.shape[0]
        
        outputs = torch.zeros(max_len, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        
        # encoder_output: [seq_len, batch, hidden_size]
        # hidden: [1, batch, hidden_size]
        encoder_output, hidden = self.encoder(src, lengths)
        
        mean = self.mean(encoder_output) # L x B x K
        logvar = self.logvar(encoder_output)
        z = self.reparameterize(mean, logvar)
        encoder_output = self.latent2hidden(z) # L x B x H
        
        hidden = hidden[-self.utter_n_layer:]
        output = tgt[0, :]
        
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_output)
                outputs[t] = output
                output = tgt[t] # [max_len, batch, output_size]
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_output)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        
        p_mean, p_var = self.compute_prior(encoder_output)
        q_mean, q_var = self.compute_posterior(mean, logvar)
        kld = self.compute_kld(p_mean, p_var, q_mean, q_var) # B
        kld = torch.mean(kld)
        
        return outputs, kld
    
    def predict(self, src, maxlen, lengths, loss=True):
        with torch.no_grad():
            batch_size = src.shape[1]
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs.cuda()
                floss = floss.cuda()

            encoder_output, hidden = self.encoder(src, lengths)
            
            mean = self.mean(encoder_output)
            encoder_output = self.latent2hidden(mean)
            
            hidden = hidden[-self.utter_n_layer:]
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output.cuda()

            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, encoder_output)
                floss[t] = output
                # output = torch.max(output, 1)[1]    # [1]
                output = output.topk(1)[1].squeeze()
                outputs[t] = output    # output: [1, output_size]

            if loss:
                return outputs, floss
            else:
                return outputs 
            
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.using_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def compute_prior(self, enc_output):
        """
        local prior p(z|x) = N(mu(x), K(x, x'))
        
        enc_output - L x B x H
        """
        l, b, h = list(enc_output.size())
        mean = enc_output.sum(dim=2) # L x B
        mean = mean.transpose(0, 1)  # B x L
        var = torch.zeros((b, l, l), requires_grad=False) # B x L x L
        if self.using_cuda: var = var.cuda()
        for i in range(l):
            for j in range(l):
                var[:, i, j] = self.kernel_func(enc_output[i,:,:], enc_output[j,:,:])
        return mean, var
    
    def kernel_func(self, x, y):
        """
        x, y - B x H
        """
        cov_xy = self.kernel_v * torch.exp(-0.5 * torch.sum(torch.pow((x - y)/self.kernel_r, 2), dim=1))
        return cov_xy
    
    def compute_posterior(self, mean, logvar):
        """
        variational posterior q(z|x) = N(mu(x), f(x)*f(x))
        
        mean, logvar - L x B x K
        """
        mean = mean.sum(dim=2) # L x B
        mean = mean.transpose(0, 1)  # B x L
        x_var = torch.exp(logvar).sum(dim=2) # L x B
        x_var = x_var.transpose(0, 1)  # B x L
        
        var_batch = []
        for b in range(mean.size(0)):
            identity_matrix = torch.eye(x_var.size(1))
            if self.using_cuda: identity_matrix = identity_matrix.cuda()
            var_batch.append(x_var[b]*identity_matrix)
        var = torch.stack(var_batch, dim=0) # B x L x L
        return mean, var
        
    def compute_kld(self, p_mean, p_var, q_mean, q_var):
        k = p_var.size(1)
        
        log_det = torch.logdet(p_var) - torch.logdet(q_var) 
        if torch.isnan(log_det).int().sum() > 0:
            if torch.isnan(q_var).int().sum() > 0:
                print('q_var has nan!!!')
                print(q_var)
        
        try:
            p_var_inv = torch.inverse(p_var) # B x L x L
            trace_batch = torch.matmul(p_var_inv, q_var) # B x L x L
            trace_list = [torch.trace(trace_batch[i]) for i in range(trace_batch.size(0))]
            trace = torch.stack(trace_list, dim=0) # B
            
            mean_diff = p_mean.unsqueeze(2) - q_mean.unsqueeze(2) # B x L x 1
            mean = torch.matmul(torch.matmul(mean_diff.transpose(1,2), p_var_inv), mean_diff) # B x 1 x 1
            
            kld = log_det - k + trace + mean.squeeze()
            kld = 0.5 * kld # B
        except:
            zeros = torch.zeros(p_mean.size(0))
            if self.using_cuda: zeros = zeros.cuda()
            kld = zeros
        return kld

if __name__ == "__main__":
    pass
