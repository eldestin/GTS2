from torch.cuda import amp
import torch
from torch import nn
import sys
sys.path.append("..")
from utils import *
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP,self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim, bias = True)
        self.layer2 = nn.Linear(hidden_dim, output_dim, bias = True)
        self.act = nn.ReLU()
    
    def forward(self, input_):
        with amp.autocast():
            return self.layer2(self.act(self.layer1(input_)))
    


# ## Test layer

# In[8]:


class Pooling_layer(nn.Module):
    def __init__(self, mode = "avg"):
        '''
        This class is a basic pooling layer on seq_len dimension
        '''
        super(Pooling_layer, self).__init__()
    
    def forward(self,last_hidden_state,head_text_idx, relation_text_idx, tail_text_idx):
        # Given last_hidden state shape (bs, seq_len, dim)
        # index shape (bs, 2)
        # return each hidden state result
        # create three tensor to store result
        with amp.autocast():
            head_pooling_state_batch  = torch.zeros(size = (last_hidden_state.shape[0], last_hidden_state.shape[-1]), device = last_hidden_state.device)
            rel_pooling_state_batch   = torch.zeros(size = (last_hidden_state.shape[0], last_hidden_state.shape[-1]), device = last_hidden_state.device)
            tail_pooling_state_batch  = torch.zeros(size = (last_hidden_state.shape[0], last_hidden_state.shape[-1]), device = last_hidden_state.device)
            for i, state in enumerate(last_hidden_state):
                # -------- First get hidden state for each text -------
                cls_hidden_state      = state[head_text_idx[i][0]].unsqueeze(0)
                #print("cls_hidden_state, ", cls_hidden_state.shape)
                head_hidden_state     = torch.mean(state[head_text_idx[i][0]: head_text_idx[i][1]+1], 0)
                relation_hidden_state = torch.mean(torch.cat([cls_hidden_state, 
                                                              state[relation_text_idx[i][0]: relation_text_idx[i][1]+1]], 
                                                             0),0)
                tail_hidden_state     = torch.mean(torch.cat([cls_hidden_state, 
                                                          state[tail_text_idx[i][0]: tail_text_idx[i][1]+1]], 
                                                         0), 0)
                head_pooling_state_batch[i] = head_hidden_state
                rel_pooling_state_batch[i]  = relation_hidden_state
                tail_pooling_state_batch[i] = tail_hidden_state
            
            return head_pooling_state_batch, rel_pooling_state_batch, tail_pooling_state_batch


# In[9]:


class KGBERT(nn.Module):
    def __init__(self,num_class, divide = True ,path = None):
        '''
        init function:
            path: pretrained model from huggingface path, if not, download it from website
        '''
        super().__init__()
        if path is not None:
            self.model = BertModel.from_pretrained(path)
        else:
            self.model = BertModel.from_pretrained('./bert-base-uncased')
        self.divide = divide
        if self.divide:
            self.pooling = Pooling_layer()
    def forward(self, input_ids,segment_ids, attention_mask, head_ids = None, rel_ids = None, tail_ids = None):
        with amp.autocast():
            if not self.divide:
                bert_out = self.model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = segment_ids)
                return bert_out
            #print(self.model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = segment_ids))
            #raise RuntimeError()
            bert_out = self.model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = segment_ids)["last_hidden_state"]
            cls_state = bert_out[:, 0, :]
            head_pooling_state, rel_pooling_state, tail_pooling_state = self.pooling(bert_out, head_ids, rel_ids, tail_ids)
            return head_pooling_state, rel_pooling_state, tail_pooling_state, cls_state


class Dot_product_attention(nn.Module):
    def __init__(self, dropout_rate):
        super(Dot_product_attention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_weight = None
    def forward(self, q,k,v, masked = True):
        '''
        q,k,v size: [N, T, F]
        '''
        res = torch.bmm(q, k.transpose(1,2)) / math.sqrt(q.shape[-1])
        # [N,T,T]
        self.attention_weight = masked_softmax(res, masked)
        return torch.bmm(self.dropout(self.attention_weight),v)

class Multi_head_attention(nn.Module):
    def __init__(self, key_size, value_size, query_size, hidden_size, num_heads,
                dropout_rate, bias = False):
        super(Multi_head_attention, self).__init__()
        self.num_heads = num_heads
        self.attention = Dot_product_attention(dropout_rate)
        
        self.W_q = nn.Linear(query_size, hidden_size, bias = bias)
        self.W_k = nn.Linear(key_size, hidden_size, bias = bias)
        self.W_v = nn.Linear(value_size, hidden_size, bias = bias)
        
        self.W_o = nn.Linear(hidden_size, hidden_size, bias = bias)
    def forward(self, q,k,v, masked = True):
        # [NH,T,hidden/H]
        q = multihead(self.W_q(q), self.num_heads)
        k = multihead(self.W_k(k), self.num_heads)
        v = multihead(self.W_v(v), self.num_heads)
        # [NH,T,T]
        output = self.attention(q,k,v, masked)  
        concat = multihead(output, self.num_heads, reverse= True)
        return self.W_o(concat)
        
class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class Add_norm(nn.Module):
        def __init__(self, norm_shape, dropout_rate, **kwargs):
            super(Add_norm, self).__init__(**kwargs)
            self.dropout = nn.Dropout(dropout_rate)
            self.layer_norm = nn.LayerNorm(norm_shape)                                
        def forward(self, X, Y):
            return self.layer_norm(self.dropout(Y) + X)
class Co_att_block(nn.Module):
    def __init__(self, ks, vs, qs, out_dim,num_head,drop_rate, num_input, num_hiddens, num_output, norm_shape):
         super(Co_att_block, self).__init__()
         self.att   = Multi_head_attention(ks,vs,qs,
                            out_dim,num_head,drop_rate)
         self.add_norm = Add_norm(norm_shape, drop_rate)
         # project text to structure
         self.FFN = MLP(num_input, num_hiddens, num_output)
         self.add_norm2 = Add_norm(norm_shape, drop_rate)
    def forward(self, q,k,v):
        # query as residual
        if len(q.shape) == 2:
            q = q.unsqueeze(1)
        #print("Q shape", q.shape)
        #print("k,v shape", k.shape)
        res1 = self.add_norm(q,self.att(q,k,v, masked = False))
        #print("Norm out shape:", res1.shape)
        res2 = self.add_norm2(res1, self.FFN(res1))
        return res2
        
        
class Score_func(nn.Module):
    """
        Func:
            Contain all the score functions we often meet. Now we finished ConvE, TransE, TransH, DisMult
        
        Args:
            sub_emb: the head embedding (subject)
            rel_emb: the relation embedding (relation)
            obj_emb: the tail embedding (object)
            kernel_size: a tuple. Only when the score function is ConvE, we need it to do the 
                        convolutional computation. i.e. kernel_size = (hight, width)
            func_type: a string indicating the score function we wanna use. default to be "TransE"
            conv_drop: a list containing floats, indicating the dropout rate we will use in the ConvE. 
                        If None, set to be all the same as "dropout" value. Default to all be the 
                        tuned parameter in compGCN paper.
            conv_bias: whether to use bias. Default to be True
            gamma: a float - margin hyperparameter. Only when we use TransE as our score function, we 
                    need it. Default to be 40.0, the tuned best parameter in compGCN.
    """
    
    def __init__(self, sub_emb, rel_emb, obj_emb, func_type="transE", 
                 kernel_size = None, conv_drop=(0.2, 0.3, 0.2), 
                 conv_bias=True, gamma=40.0):
        # we can't use self.__class__, because it may cause a recursive problem
        super(Score_func, self).__init__()
        
        self.func_type = func_type.lower()
        self.gamma     = gamma
        self.sub_emb   = sub_emb
        self.rel_emb   = rel_emb
        self.obj_emb   = obj_emb
        
        if self.func_type == "transh":
            self.relation_norm_embedding  = torch.nn.Embedding(num_embeddings=relation_num,
                                                              embedding_dim=self.dimension)
            self.relation_hyper_embedding = torch.nn.Embedding(num_embeddings=relation_num,
                                                               embedding_dim=self.dimension)
            self.entity_embedding         = torch.nn.Embedding(num_embeddings=entity_num,
                                                               embedding_dim=self.dimension)
        
        if self.func_type == "conve":
            assert not kernel_size is None  # to ensure that the kernel size is defined
            
            if not conv_drop:
                self.hidden_drop = [dropout, dropout, dropout]
            else:
                l = len(hidden_drop)
                assert l <= 3  # ensure the length of hidden_drop smaller equal to 3
                if l == 1:
                    self.conv_drop = [conv_drop[0], conv_drop[0], conv_drop[0]]
                elif l == 2:
                    self.conv_drop = [conv_drop[0], conv_drop[0], conv_drop[1]]
                else:
                    self.conv_drop = conv_drop
                
            
            self.kernel_size    = kernel_size
            self.bias           = conv_bias
            
            self.bn0            = torch.nn.BatchNorm2d(1)
            self.bn1            = torch.nn.BatchNorm2d(self.out_channels)
            self.bn2            = torch.nn.BatchNorm1d(self.kernel_size)

            self.hidden_drop    = torch.nn.Dropout(self.conv_drop[0])
            self.hidden_drop2   = torch.nn.Dropout(self.conv_drop[1])
            self.feature_drop   = torch.nn.Dropout(self.conv_drop[2])
            self.m_conv1        = torch.nn.Conv2d(1, out_channels=self.out_channels, 
                                                  kernel_size=(self.kernel_size, self.kernel_size), 
                                                  stride=1, padding=0, bias=self.bias)

            flat_sz_h           = int(2*self.kernel_size[1]) - self.kernel_size + 1
            flat_sz_w           = self.kernel_size[0] - self.kernel_size + 1
            self.flat_sz        = flat_sz_h * flat_sz_w * self.out_channels
            self.fc             = torch.nn.Linear(self.flat_sz, self.kernel_size)
    
    def concat(self, e1_embed, rel_embed):
        e1_embed    = e1_embed. view(-1, 1, self.p.embed_dim)
        rel_embed   = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp   = torch.cat([e1_embed, rel_embed], 1)
        stack_inp   = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
        return stack_inp
    
    def projected(self, ent, norm):
        norm = F.normalize(norm, p=2, dim=-1)
        return ent - torch.sum(ent * norm, dim = 1, keepdim=True) * norm
    
    def forward_score(self):
        with amp.autocast():
            if   self.func_type == "transe":
                x        = self.sub_emb + self.rel_emb - self.obj_emb
            elif self.func_type == "transh":
                head       = self.entity_embedding(self.sub_emb)
                tail       = self.entity_embedding(self.obj_emb)
                r_norm     = self.relation_norm_embedding(self.rel_emb)
                r_hyper    = self.relation_hyper_embedding(self.rel_emb)
                head_hyper = self.projected(head, r_norm)
                tail_hyper = self.projected(tail, r_norm)
                x          = torch.norm(head_hyper + r_hyper - tail_hyper, p=2, dim=2)
            elif self.func_type == "distmult":
                x        =   torch.mm(self.sub_emb + self.rel_emb, self.obj_emb.transpose(1, 0))
                x        +=  self.bias.expand_as(x)
            elif self.func_type == "conve":
                stk_inp  = self.concat(sub_emb, rel_emb)
                x        = self.bn0(stk_inp)
                x        = self.m_conv1(x)
                x        = self.bn1(x)
                x        = F.relu(x)
                x        = self.feature_drop(x)
                x        = x.view(-1, self.flat_sz)
                x        = self.fc(x)
                x        = self.hidden_drop2(x)
                x        = self.bn2(x)
                x        = F.relu(x)

                x = torch.mm(x, self.obj_emb.transpose(1,0))
                x += self.bias.expand_as(x)
            return x
