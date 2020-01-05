
from __future__ import division
from __future__ import print_function
from builtins import range

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable
import ecc


class RNNGraphConvModule(nn.Module):
    """
    Computes recurrent graph convolution using filter weights obtained from a Filter generating network (`filter_net`).
    Its result is passed to RNN `cell` and the process is repeated over `nrepeats` iterations.
    Weight sharing over iterations is done both in RNN cell and in Filter generating network.
    """
    def __init__(self, cell, filter_net, gc_info=None, nrepeats=1, cat_all=False, edge_mem_limit=1e20):
        super(RNNGraphConvModule, self).__init__()
        self._cell = cell
        self._isLSTM = 'LSTM' in type(cell).__name__
        self._isRNN= 'RNN' in type(cell).__name__
        self._fnet = filter_net
        self._nrepeats = nrepeats
        self._cat_all = cat_all
        self._edge_mem_limit = edge_mem_limit
        self.set_info(gc_info)

    def set_info(self, gc_info):
        self._gci = gc_info

    def forward(self, hx):
        # get graph structure information tensors
        idxn, idxe, degs, degs_gpu, edgefeats = self._gci.get_buffers()
        edgefeats = Variable(edgefeats, requires_grad=False)

        # evalute and reshape filter weights (shared among RNN iterations)
        weights = self._fnet(edgefeats)
        nc = hx.size(1)
        assert hx.dim()==2 and weights.dim()==2 and weights.size(1) in [nc, nc*nc]
        if weights.size(1) != nc:
            weights = weights.view(-1, nc, nc)

        # repeatedly evaluate RNN cell
        hxs = [hx]
        if self._isLSTM or self._isRNN:
            cx = Variable(hx.data.new(hx.size()).fill_(0))
     
            
            
#==============================================================================
        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx, weights)
        if self._isLSTM: #or self._isRNN
             hx1, cx1 = self._cell(input, (hx, cx))
        else:
             hx1 = self._cell(input, hx)
        hxs.append(hx1)
         
        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx1, weights)
        if self._isLSTM:
             hx2, cx2 = self._cell(input, (hx1, cx1))
        else:
             hx2 = self._cell(input, hx1)
        hxs.append(hx2)
         
        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx2, weights)
        if self._isLSTM:
             hx3, cx3 = self._cell(input, (hx2, cx2))
        else:
             hx3 = self._cell(input, hx2)
#        hxs.append(hx3)
        skipCon1 = hx1+hx3
        hxs.append(skipCon1)
         
         
         
        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(skipCon1, weights)
        if self._isLSTM:
             hx4, cx4 = self._cell(input, (skipCon1, cx3))
        else:
             hx4 = self._cell(input, skipCon1)
        hxs.append(hx4)
         
        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx4, weights)
        if self._isLSTM:
             hx5, cx5 = self._cell(input, (hx4, cx4))
        else:
             hx5 = self._cell(input, hx4)
#        hxs.append(hx5)
        skipCon2 = hx3+hx5
        hxs.append(skipCon2)
         
        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(skipCon2, weights)
        if self._isLSTM:
             hx6, cx6 = self._cell(input, (skipCon2, cx5))
        else:
             hx6 = self._cell(input, skipCon2)
        hxs.append(hx6)
         
        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx6, weights)
        if self._isLSTM:
             hx7, cx7 = self._cell(input, (hx6, cx6))
        else:
             hx7 = self._cell(input, hx6)
#        hxs.append(hx7)
        skipCon3 = hx5+hx7
        hxs.append(skipCon3)
         
        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(skipCon3, weights)
        if self._isLSTM:
             hx8, cx8 = self._cell(input, (skipCon3, cx7))
        else:
             hx8 = self._cell(input, skipCon3)
        hxs.append(hx8)
         
        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx8, weights)
        if self._isLSTM:
             hx9, cx9 = self._cell(input, (hx8, cx8))
        else:
             hx9 = self._cell(input, hx8)
       #  hxs.append(hx9)
        skipCon4 = hx7+hx9
        hxs.append(skipCon4)
         
        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(skipCon4, weights)
        if self._isLSTM:
             hx10, cx10 = self._cell(input, (skipCon4, cx9))
        else:
             hx10 = self._cell(input, skipCon4)
        hxs.append(hx10)    
             
#            other 10 added
#        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx10, weights)
#        if self._isLSTM:
#             hx11, cx11 = self._cell(input, (hx10, cx10))
#        else:
#             hx11 = self._cell(input, hx10)
#       
#        skipCon5 = hx9+hx11
#        hxs.append(skipCon5)
#        
#        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(skipCon5, weights)
#        if self._isLSTM:
#             hx12, cx12 = self._cell(input, (skipCon5, cx11))
#        else:
#             hx12 = self._cell(input, skipCon5)
#        hxs.append(hx12)
#        
#        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx12, weights)
#        if self._isLSTM:
#             hx13, cx13 = self._cell(input, (hx12, cx12))
#        else:
#             hx13 = self._cell(input, hx12)
#       
#        skipCon6 = hx11+hx13
#        hxs.append(skipCon6)
#        
#        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(skipCon6, weights)
#        if self._isLSTM:
#             hx14, cx14 = self._cell(input, (skipCon6, cx13))
#        else:
#             hx14 = self._cell(input, skipCon6)
#        hxs.append(hx14)
#          
#        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx14, weights)
#        if self._isLSTM:
#             hx15, cx15 = self._cell(input, (hx14, cx14))
#        else:
#             hx15 = self._cell(input, hx14)
#       
#        skipCon7 = hx13+hx15
#        hxs.append(skipCon7)
#        
#        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(skipCon7, weights)
#        if self._isLSTM:
#             hx16, cx16 = self._cell(input, (skipCon7, cx15))
#        else:
#             hx16 = self._cell(input, skipCon7)
#        hxs.append(hx16)
        
#        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx16, weights)
#        if self._isLSTM:
#             hx17, cx17 = self._cell(input, (hx16, cx16))
#        else:
#             hx17 = self._cell(input, hx16)
#       
#        skipCon8 = hx15+hx17
#        hxs.append(skipCon8)
#        
#        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(skipCon8, weights)
#        if self._isLSTM:
#             hx18, cx18 = self._cell(input, (skipCon8, cx17))
#        else:
#             hx18 = self._cell(input, skipCon8)
#        hxs.append(hx18)   
#        
#        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(hx18, weights)
#        if self._isLSTM:
#             hx19, cx19 = self._cell(input, (hx18, cx18))
#        else:
#             hx19 = self._cell(input, hx18)
#       
#        skipCon9 = hx17+hx19
#        hxs.append(skipCon9)
#        
#        input = ecc.GraphConvFunction(nc, nc, idxn, idxe, degs, degs_gpu, self._edge_mem_limit)(skipCon9, weights)
#        if self._isLSTM:
#             hx20, cx20 = self._cell(input, (skipCon9, cx19))
#        else:
#             hx20 = self._cell(input, skipCon9)
#        hxs.append(hx20) 
#==============================================================================

        return torch.cat(hxs,1) if self._cat_all else hx


class ECC_CRFModule(nn.Module):
    """
    Adapted "Conditional Random Fields as Recurrent Neural Networks" (https://arxiv.org/abs/1502.03240)
    `propagation` should be ECC with Filter generating network producing 2D matrix.
    """
    def __init__(self, propagation, nrepeats=1):
        super(ECC_CRFModule, self).__init__()
        self._propagation = propagation
        self._nrepeats = nrepeats

    def forward(self, input):
        Q = nnf.softmax(input)
        for i in range(self._nrepeats):
            Q = self._propagation(Q) # todo: speedup possible by sharing computation of fnet
            Q = input - Q
            if i < self._nrepeats-1:
                Q = nnf.softmax(Q) # last softmax will be part of cross-entropy loss
        return Q
 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, layernorm=True, ingate=True, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.n_layers = n_layers
        self._layernorm = layernorm
        self._ingate = ingate
        self.rnn =  nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh') #, bias=True, nonlinearity='tanh'

    def forward(self, input, hidden):
   
        output = self.rnn(input, hidden)
     
        return output

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        

class GRUCellEx(nn.GRUCell):
    """ Usual GRU cell extended with layer normalization and input gate.
    """
    def __init__(self, input_size, hidden_size, bias=True, layernorm=True, ingate=True):
        super(GRUCellEx, self).__init__(input_size, hidden_size, bias)
        self._layernorm = layernorm
        self._ingate = ingate
        if layernorm:
            self.add_module('ini', nn.InstanceNorm1d(1, eps=1e-5, affine=False))
            self.add_module('inh', nn.InstanceNorm1d(1, eps=1e-5, affine=False))
        if ingate:
            self.add_module('ig', nn.Linear(hidden_size, input_size, bias=True))

    def _normalize(self, gi, gh):
        if self._layernorm: # layernorm on input&hidden, as in https://arxiv.org/abs/1607.06450 (Layer Normalization)
            gi = self._modules['ini'](gi.unsqueeze(1)).squeeze(1)
            gh = self._modules['inh'](gh.unsqueeze(1)).squeeze(1)
        return gi, gh

    def forward(self, input, hidden):
#        print(self.state_dict())
#        exit()   
       
        
        if self._ingate:
            input = nnf.sigmoid(self._modules['ig'](hidden)) * input

        # GRUCell in https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py extended with layer normalization
        if input.is_cuda:
            gi = nnf.linear(input, self.weight_ih)
            gh = nnf.linear(hidden, self.weight_hh)
            gi, gh = self._normalize(gi, gh)
            state = torch.nn._functions.thnn.rnnFusedPointwise.GRUFused
            try: #pytorch >=0.3
                return state.apply(gi, gh, hidden) if self.bias_ih is None else state.apply(gi, gh, hidden, self.bias_ih, self.bias_hh)
            except: #pytorch <=0.2
                return state()(gi, gh, hidden) if self.bias_ih is None else state()(gi, gh, hidden, self.bias_ih, self.bias_hh)

        gi = nnf.linear(input, self.weight_ih, self.bias_ih)
        gh = nnf.linear(hidden, self.weight_hh, self.bias_hh)
        gi, gh = self._normalize(gi, gh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = nnf.sigmoid(i_r + h_r)
        inputgate = nnf.sigmoid(i_i + h_i)
        newgate = nnf.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def __repr__(self):
        s = super(GRUCellEx, self).__repr__() + '('
        if self._ingate:
            s += 'ingate'
        if self._layernorm:
            s += ' layernorm'
        return s + ')'


class LSTMCellEx(nn.LSTMCell):
    """ Usual LSTM cell extended with layer normalization and input gate.
    """
    def __init__(self, input_size, hidden_size, bias=True, layernorm=True, ingate=True):
        super(LSTMCellEx, self).__init__(input_size, hidden_size, bias)
        self._layernorm = layernorm
        self._ingate = ingate
        if layernorm:
            self.add_module('ini', nn.InstanceNorm1d(1, eps=1e-5, affine=False))
            self.add_module('inh', nn.InstanceNorm1d(1, eps=1e-5, affine=False))
        if ingate:
            self.add_module('ig', nn.Linear(hidden_size, input_size, bias=True))

    def _normalize(self, gi, gh):
        if self._layernorm: # layernorm on input&hidden, as in https://arxiv.org/abs/1607.06450 (Layer Normalization)
            gi = self._modules['ini'](gi.unsqueeze(1)).squeeze(1)
            gh = self._modules['inh'](gh.unsqueeze(1)).squeeze(1)
        return gi, gh

    def forward(self, input, hidden):

        
        if self._ingate:
            input = nnf.sigmoid(self._modules['ig'](hidden[0])) * input

        # GRUCell in https://github.com/pytorch/pytorch/blob/master/torch/nn/_functions/rnn.py extended with layer normalization
        if input.is_cuda:
            gi = nnf.linear(input, self.weight_ih)
            gh = nnf.linear(hidden[0], self.weight_hh)
            gi, gh = self._normalize(gi, gh)
            state = torch.nn._functions.thnn.rnnFusedPointwise.LSTMFused
            try: #pytorch >=0.3
                return state.apply(gi, gh, hidden[1]) if self.bias_ih is None else state.apply(gi, gh, hidden[1], self.bias_ih, self.bias_hh)
            except: #pytorch <=0.2
                return state()(gi, gh, hidden[1]) if self.bias_ih is None else state()(gi, gh, hidden[1], self.bias_ih, self.bias_hh)

        gi = nnf.linear(input, self.weight_ih, self.bias_ih)
        gh = nnf.linear(hidden[0], self.weight_hh, self.bias_hh)
        gi, gh = self._normalize(gi, gh)

        ingate, forgetgate, cellgate, outgate = (gi+gh).chunk(4, 1)
        ingate = nnf.sigmoid(ingate)
        forgetgate = nnf.sigmoid(forgetgate)
        cellgate = nnf.tanh(cellgate)
        outgate = nnf.sigmoid(outgate)

        cy = (forgetgate * hidden[1]) + (ingate * cellgate)
        hy = outgate * nnf.tanh(cy)
        return hy, cy

    def __repr__(self):
        s = super(LSTMCellEx, self).__repr__() + '('
        if self._ingate:
            s += 'ingate'
        if self._layernorm:
            s += ' layernorm'
        return s + ')'