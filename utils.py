import numpy as np
import math
import pandas as pd
import torch
from torch import nn
import torch_scatter
import inspect
from transformers import BertTokenizerFast, BertModel

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2 as irfft
    from torch.fft import rfft2  as rfft
import ast
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc

def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param
def com_mult(a, b):
    r1, i1 = a[:, 0], a[:, 1]
    r2, i2 = b[:, 0], b[:, 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)
    
def conj(a): 
    #print("in conj", a.shape)
    a[:, 1] = -a[:, 1]
    return a
def ccorr(a, b):
    #print("in ccorr", a.shape)
    return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a)), torch.fft.rfft(b)),a.shape[-1])
    
    
def masked_softmax(attention_score, masked = True):
    '''
    Input:
        1. attention: [N*H,T,T]
    RETURN:
        1. LOWER triangular, [NH,T,T]
    '''
    if masked:
        nh   = attention_score.shape[0]
        ones = torch.ones_like(attention_score[0,:,:])
        tril = torch.tril(ones).unsqueeze(0)
        mask = torch.repeat_interleave(tril,nh,0)
        pad  = torch.ones_like(mask)*(-1e6)
        out  = torch.where(mask == 0,pad,attention_score)
        return torch.softmax(out, -1)
    else:
        return torch.softmax(attention_score, dim = -1)

    
def multihead(input_, num_heads, reverse = False):
    if not reverse:
        # (N, T, hidden)
        # change to (N,T, head, hidden/head)
        input_ = input_.reshape(input_.shape[0], input_.shape[1], num_heads, -1)
        input_ = input_.permute(0,2,1,3)
        # (N*h, T, hidden/head)
        return input_.reshape(-1, input_.shape[2], input_.shape[3])
    else:
        # change back to (N, head, T, hidden/head)
        input_ = input_.reshape(-1, num_heads, input_.shape[1], input_.shape[2])
        input_ = input_.permute(0,2,1,3)
        return input_.reshape(input_.shape[0], input_.shape[1], -1)
        
        
def change_input(tokenizer, text1, text2=None, text3=None, labels = None,max_length=512):
    '''
    This function will change the given input from double to triple
    '''
    #do the basic tokenization without changing to index
    id_ls = []
    #print(text3)
    tokens_1 = tokenizer.tokenize(text1)
    if text2 is not None:
        tokens_2 = tokenizer.tokenize(text2)
    if text3 is not None:
        tokens_3 = tokenizer.tokenize(text3)
    #as shown in kg-bert, do the truncation
    while True:
        #do the trunctation 
        total_length = len(tokens_1)+len(tokens_2)+len(tokens_3)
        if total_length<= max_length-4:
            break
        if len(tokens_1)>len(tokens_2) and len(tokens_1)>len(tokens_3):
            tokens_1.pop()
        elif len(tokens_2)>len(tokens_1) and len(tokens_2)>len(tokens_3):
            tokens_2.pop()
        elif len(tokens_3)>len(tokens_2) and len(tokens_3)>len(tokens_1):
            tokens_3.pop()
        else:
            #else pop the token3(tail)
            tokens_3.pop()
    #segment encoding
    tokens_1_idx_start = 0
    tokens_1_idx_end = len(tokens_1) + 1
    final_token = ["[CLS]"]+tokens_1+["[SEP]"]
    #segment for first sentence
    segment_ids = [0]*len(final_token)
    if text2 is not None:
        final_token+=tokens_2+["[SEP]"]
        segment_ids+=[1]*(len(tokens_2)+1)
        tokens_2_idx_start = tokens_1_idx_end + 1
        tokens_2_idx_end   = tokens_2_idx_start + len(tokens_2) 
    if text3 is not None:
        final_token+=tokens_3+["[SEP]"]
        segment_ids+=[0]*(len(tokens_3)+1)
        tokens_3_idx_start = tokens_2_idx_end + 1
        tokens_3_idx_end   = tokens_3_idx_start + len(tokens_3) 
    #change it to the index
    input_ids = tokenizer.convert_tokens_to_ids(final_token)
    #for padding
    padding = [0]*(max_length - len(input_ids))
    #for attention mask
    attention_mask = [1]*len(input_ids)
    input_ids+=padding
    attention_mask+= padding
    segment_ids+=padding
    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length
    assert len(segment_ids) == max_length
    return {"input_ids": input_ids,
            "segment_ids": segment_ids,
            "attention_mask": attention_mask,
            "labels":labels,
            "head_idx": (tokens_1_idx_start, tokens_1_idx_end),
            "relation_idx":(tokens_2_idx_start, tokens_2_idx_end),
            "tail_idx": (tokens_3_idx_start, tokens_3_idx_end)
    }

def write_log(info, name):
    try:
        with open(name,"a") as f:
            f.write(info+"\n")
    except FileNotFoundError:
        with open(name,"w") as f:
            f.write(info+"\n")



def get_result(x):
    x[x>=0.5] = int(1)
    x[x<0.5] = int(0)
    return x
    
def freeze_params(net, layer_names):
    '''
    Support two types of layer when freezing parameters
    params:
        1. net: network
        2. layer_name: a list containing layer's name
    '''
    for name, param in net.named_parameters():
        for l_name in layer_names:
            if l_name in name:
                print("Freezing layer: " + name)
                write_log("Freezing layer: " + name,"train_log")
                param.requires_grad = False

def unfreeze_params(net,layer_names):
    '''
    Support two types of layer when unfreezing parameters
    params:
        1. net: network
        2. layer_name1: First type layer
    '''
    for name, param in net.named_parameters():
        for l_name in layer_names:
            if l_name in name:
                print("UnFreezing layer: " + name)
                write_log("UnFreezing layer: " + name,"train_log")
                param.requires_grad = True



def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device(i):
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
    

def read_dataset(path, df_path, final_year, wiki = True):
    '''
    Return graph structure, train df and language model.
    '''
    if wiki:
        wiki = "data/wiki/wiki_"
    else:
        wiki = "data/Yago/"
    graph_info   = np.load(path + "./"+wiki+"%d_graph_info.npy"%final_year, allow_pickle=True).flatten()[0]

    num_nodes    = [torch.tensor(i) for i in graph_info["num_ent"]]
    num_relation = [torch.tensor(i) for i in graph_info["num_rel"]]
    # print(type(graph_info["adj"]))
    # print(type(graph_info["adj"][0]))
    # print(type(graph_info["adj"][0][0]))
    # print(type(graph_info["adj"][0][0][0]))
    edge_idx     = [torch.tensor(np.array([np.array(j) for j in i])) for i in graph_info["adj"]]
    tmp          = [np.array([np.array(j) for j in i]) for i in graph_info["adj"]]
    # print(graph_info["num_ent"])
    # print(graph_info["num_rel"])
    # print(tmp[0].shape)
    # print([np.min(np.min(tmp[i])) for i in range(len(tmp))])
    # print([np.max(np.max(tmp[i])) for i in range(len(tmp))])
    edge_type    = [torch.tensor(i) for i in graph_info["edge_type"]]
    head2idx = np.load(path + "./"+wiki+'ent_dic.npy', allow_pickle=True).item()
    rel2idx =  np.load(path + "./"+wiki+'rel_dic.npy', allow_pickle=True).item()
    train = pd.read_csv(df_path + "./"+wiki+"train_%d.csv"%final_year).drop("Unnamed: 0", axis = 1)
    train["index_where"] = train["index_where"].apply(ast.literal_eval)
    return edge_idx, edge_type, head2idx, rel2idx, train, len(rel2idx), len(head2idx)   # num_relation[-1], num_nodes[-1]
