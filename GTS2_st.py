import torch
from torch import nn
from utils import *
from Layer.R_GCN_Block import *
from Layer.Utils_Block import *
from torch.cuda import amp

class CompGcn_with_temporal_bert(nn.Module):
    def __init__(self, conv_dim, num_relation, num_entity,node_dim, num_hiddens ,num_basis_vector, num_class,num_heads = 2,time_stamp = 2,
                 num_layers = 2,score_func="TransE", divide = True, comp_type = "sub", fusion_dim = 768, module = None, norm_shape = 64, seq_len = 3):
        '''
        Notice that in preprocessing, we assume that the node number will not be changed on the graph, only change relation.
        input params:
            1. conv_dim, a list of tuple, [(channel 1, channel2, channel 3), (channel 3, channel 4, channel 5),...]
            2. num_layer, number of CompGCN layer, now assume to 2 per graph
            3. num_relation, the number of relations_type for each graph(after preprocessing, should be the same for each graph)
            4. num_entity, number of entity, should be the same for each graph
            5. node_dimension, dimension of nodes
            6. num_basis_vector, the first layer basis of first graph.
            7. edge_idx, a list of edge_idx
            8. edge_type, a list of edge_type
            9. num_class, classification number 
            10. time stamp: How many time steps 
            11. divide: whether divide into head, rel, tail embd, if False, will use [cls] as total embedding
        '''
        assert len(conv_dim) == time_stamp, "time stamp length should be the same as number of convloution dimension list!, got time stamp "+str(time_stamp)+" with conv_dim "+str(len(conv_dim))
        super(CompGcn_with_temporal_bert,self).__init__()
        self.divide = divide
        # extract head, rel, tail text embd
        print("Read in Language Model: ")
        self.model = KGBERT(num_class, divide = divide)
        self.num_relation = num_relation
        self.conv_dim = conv_dim
        self.node_features = get_param(shape= (num_entity,node_dim))
        assert node_dim == conv_dim[0][0]
        self.temporal_blk = nn.Sequential()
        self.drop_bert = nn.Dropout(0.1)
        print("Initialize Graph Block: ")
        for i in range(time_stamp):
            if i == 0:
                self.temporal_blk.add_module("Temporal block Basis" , CompGcn_total(conv_dim[i], num_relation, num_basis_vector, 
                                                                                  num_layers,True, comp_type = comp_type))
            else:
                self.temporal_blk.add_module("Temporal block" + str(i), CompGcn_total(conv_dim[i], num_relation, num_basis_vector, 
                                                                                  num_layers,False, comp_type = comp_type))
        self.mlp = nn.Linear(768+num_hiddens ,num_class)
        self.dropout_node = nn.Dropout(0.1)
        self.dropout_rel = nn.Dropout(0.1)
        self.mlp = self.func_init(self.mlp)
        self.score = score_func
        self.time_stamp = time_stamp
        self.hidden_state = num_hiddens
        self.num_node = num_entity
        # add GRU 
        self.W_xr = nn.Linear(conv_dim[0][-1],num_hiddens)
        self.W_xz = nn.Linear(conv_dim[0][-1],num_hiddens)
        self.W_xh = nn.Linear(conv_dim[0][-1],num_hiddens)
        self.W_hr = nn.Linear(num_hiddens,num_hiddens, bias = True)
        self.W_hz = nn.Linear(num_hiddens,num_hiddens, bias = True)
        self.W_hh = nn.Linear(num_hiddens,num_hiddens, bias = True)
        self.act_update = nn.Sigmoid()
        self.act_hidden = nn.Tanh()
        self.act_reset = nn.Sigmoid()
        
        # fusion model concatenation
        # project structure information dimension into textual dimension
        # or project textual dimension into structure dimension
        self.fusion_dim = fusion_dim
        self.last_hidden_dim = conv_dim[-1][-1]
        self.module = module
        # Add attention part
        self.temporal_attention = Multi_head_attention(conv_dim[-1][-1],conv_dim[-1][-1],conv_dim[-1][-1],
                                                      num_hiddens,num_heads,0.1)
        self.tmp_pos_enc = PositionalEncoding(conv_dim[-1][-1], 0.1)
        
        # semantic as 2-dim feature, use cls
        self.co_att1 = Co_att_block(conv_dim[-1][-1], conv_dim[-1][-1], 768, 768, num_heads, 0.1, 768, 768//2 ,768, 768)
        # structure as 2-dim feature, use score output
        self.co_att2 = Co_att_block(768,768,num_hiddens, num_hiddens, num_heads, 0.1, num_hiddens, num_hiddens*2, num_hiddens, norm_shape)
        # project text to structure
        # use another two Co-attention block for reduce dimension
        # query as cls, k,v as co_embd
        # query as co_embd, k,v as text
        self.textual_pos_enc = PositionalEncoding(768, 0.1)
        self.st_pos_enc = PositionalEncoding(num_hiddens, 0.1)
        self.weights = nn.Parameter(torch.ones(seq_len,1))

    def func_init(self,m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        return m
    def init_state(self, device):
        if self.module == "node":
            return torch.zeros(self.num_node, self.conv_dim[1][0], device = device)
        if self.module == "rel":
            return torch.zeros(self.num_relation *2 +1, self.conv_dim[1][0], device = device)
        else:
            return torch.zeros(self.num_node, self.conv_dim[1][0], device = device)
        
    def forward(self, input_ids,segment_ids, attention_mask  ,head_index, tail_index, rel_index, edge_idx, edge_type, 
               head_text_ids = None, rel_text_ids = None, tail_text_ids = None):
        '''
        Node index and rel index are corresponding information in a batch for bert part, we only care about the node, edge relation in a batch.
        Since the embedding is tail - relation to head, the source will be tail, target will be head
        '''
        with amp.autocast():
            device = self.node_features.device
            state  = self.init_state(device)
            hidden_state_all = torch.ones((self.node_features.shape[0],self.time_stamp,self.hidden_state), device = device)#N,T,F
            for i, blk in enumerate(self.temporal_blk):
                if i == 0:
                    #print("First block:")
                    node_embd, rel_embd = blk(self.node_features, device = device, edge_idx = edge_idx[i], edge_type = edge_type[i])
                    node_embd, rel_embd = self.dropout_node(node_embd), self.dropout_rel(rel_embd)
                    Z = self.act_update(self.W_xz(node_embd) + self.W_hz(state))
                    R = self.act_reset(self.W_xr(node_embd) + self.W_hr(state))
                    H_candidate = self.act_hidden(self.W_xh(node_embd) + self.W_hh(R * state))
                    state = Z * state + (1 - Z) * H_candidate
                    # attention on node
                    hidden_state_all[:,i,:] = state
                else:
                    #print(str(i)+"th block:")
                    if self.module == "node":
                        # GRU on node, att on node
                        # print("State shape would be: ", state.shape)
                        # print("State is: ", state)
                        node_embd, rel_embd = blk(node_embd = state, rel_embd = rel_embd,device = device, edge_idx = edge_idx[i],
                                                                             edge_type = edge_type[i])
                        node_embd, rel_embd = self.dropout_node(node_embd), self.dropout_rel(rel_embd)
                        Z = self.act_update(self.W_xz(node_embd) + self.W_hz(state))
                        R = self.act_reset(self.W_xr(node_embd) + self.W_hr(state))
                        H_candidate = self.act_hidden(self.W_xh(node_embd) + self.W_hh(R * state))
                        state = Z * state + (1 - Z) * H_candidate
                        # attention on node
                        hidden_state_all[:,i,:] = state
                    elif self.module == "rel":
                        # GRU on rel, att on node
                     #   print(rel_embd.shape, state.shape)
                        node_embd, rel_embd = blk(node_embd = node_embd, rel_embd = state ,device = device, edge_idx = edge_idx[i],
                                                                             edge_type = edge_type[i])
                        node_embd, rel_embd = self.dropout_node(node_embd), self.dropout_rel(rel_embd)
                        Z = self.act_update(self.W_xz(rel_embd) + self.W_hz(state))
                        R = self.act_reset(self.W_xr(rel_embd) + self.W_hr(state))
                        H_candidate = self.act_hidden(self.W_xh(rel_embd) + self.W_hh(R * state))
                        state = Z * state + (1 - Z) * H_candidate
                        # attention on node
                        hidden_state_all[:,i,:] = node_embd
                    else:
                        # GRU on state, att on node
                     #   print(rel_embd.shape, state.shape)
                        node_embd, rel_embd = blk(node_embd = node_embd, rel_embd = rel_embd ,device = device, edge_idx = edge_idx[i],
                                                                             edge_type = edge_type[i])
                        node_embd, rel_embd = self.dropout_node(node_embd), self.dropout_rel(rel_embd)
                        Z = self.act_update(self.W_xz(node_embd) + self.W_hz(state))
                        R = self.act_reset(self.W_xr(node_embd) + self.W_hr(state))
                        H_candidate = self.act_hidden(self.W_xh(node_embd) + self.W_hh(R * state))
                        state = Z * state + (1 - Z) * H_candidate
                        # attention on node
                        hidden_state_all[:,i,:] = state
                    

                        
                
            hidden_state_all = self.tmp_pos_enc(hidden_state_all) 
            #N,F
            node_embd = self.temporal_attention(hidden_state_all, hidden_state_all, hidden_state_all)
            node_embd = node_embd[:,-1,:]
            #B,F
            hidden_node_state = node_embd[tail_index,:]
            hidden_rel_state  = rel_embd[rel_index,:]
            hidden_target_state = node_embd[head_index,:]
            # structure key, value pair
            st_embd = torch.cat([hidden_target_state.unsqueeze(1),hidden_rel_state.unsqueeze(1) ,hidden_node_state.unsqueeze(1)], dim = 1)
            # structure query
            score    = Score_func(hidden_target_state, hidden_rel_state, hidden_node_state, func_type=self.score)
            score    = score.forward_score()
            #score    = self.ln_st(score)
            
            # semantic query
            head_text_embd, rel_text_embd, tail_text_embd, cls_state = self.model(input_ids = input_ids, attention_mask = attention_mask, segment_ids = segment_ids, 
                                                                       head_ids = head_text_ids, rel_ids = rel_text_ids, tail_ids = tail_text_ids)
            # semantic key-value pair
            textual_embd = torch.cat([head_text_embd.unsqueeze(1), rel_text_embd.unsqueeze(1), tail_text_embd.unsqueeze(1)], dim = 1)
            #textual_embd = self.ln_textual(bert_out)
            textual_embd = self.textual_pos_enc(textual_embd)
            
            st_embd = self.st_pos_enc(st_embd)
            # semantic as query, structure as k,v
            score_sem = self.co_att1(textual_embd, st_embd, st_embd)
            #print(score_sem.shape)
            # St as query, semantic as k,v
            score_st  = self.co_att2(st_embd, textual_embd, textual_embd)
            logit = torch.cat([score_sem, score_st], dim = -1)
            #print(logit.shape)
            #print(textual_embd.shape)
            logit = (logit * self.weights).sum(dim=1) / torch.abs(self.weights).sum()
            score_ffn = self.mlp(logit)
        return score_ffn 
