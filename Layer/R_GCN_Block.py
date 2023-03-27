from torch.cuda import amp
import torch
from torch import nn
import torch_scatter
try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2 as irfft
    from torch.fft import rfft2  as rfft
import sys
sys.path.append("..")
from utils import *

class CompGcnBasis(nn.Module):
    nodes_dim = 0
    head_dim = 0
    tail_dim = 1
    def __init__(self, in_channels, out_channels, num_relations, num_basis_vector,act = torch.tanh,cache = True,dropout = 0.2, comp_type = "sub"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_basis_vector = num_basis_vector
        self.act = act
        self.device = None
        self.cache = cache
        self.comp_type = comp_type
        #----------- Creating learnable basis vector , shape is (num_basis, feature_size(in channel))
        self.basis_vector = get_param((num_basis_vector, in_channels))
        # this weight matrix initialize the weight features for each relation(including inverse), shape is (2*num_relations, num_basis)
        self.rel_weight = get_param((num_relations*2, self.num_basis_vector))
        # this learnable weight matrix is for projection, that project each relation to the same dimension of node_dimension
        self.weight_rel = get_param((in_channels,out_channels))
        # add another embedding for loop
        self.loop_rel = get_param((1,in_channels))
        #----------- Creating three updated matrix, as three kind of relations updating, in, out, loop
        # using for updating weight
        self.w_in = get_param((in_channels,out_channels))
        self.w_out = get_param((in_channels,out_channels))
        self.w_loop = get_param((in_channels,out_channels))
        
        # define some helpful parameter
        self.in_norm, self.out_norm = None, None
        self.in_index, self.out_index = None, None
        self.in_type, self.out_type = None, None
        self.loop_index, self.loop_type =None, None
        
        self.drop = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_channels)
    def relation_transform(self, entity_embedding, relation_embedding,type_):
        '''
        This function given entity embedding and relation embedding, in order return three types of 
        non-parameterized operations, which is subjection, corr, multiplication
        '''
        assert type_ in ["mul","sub","corr"], "not implemented now"
        if type_ == "mul":
            out = entity_embedding*relation_embedding
        elif type_ == "sub":
            out = entity_embedding - relation_embedding
        else:
            out = ccorr(entity_embedding,relation_embedding)
        return out
    
    def normalization(self, edge_index, num_entity):
        '''
        As normal GCN, this function calculate the normalization adj matrix 
        '''
        head, tail = edge_index
        edge_weight = torch.ones_like(head).float()
        degree = torch_scatter.scatter_add(edge_weight,head,dim_size=num_entity,dim = self.nodes_dim)
        degree_inv = degree.pow(-0.5)
        # if inf, in order to prevent nan in scatter function
        degree_inv[degree_inv == float("inf")] = 0
        norm = degree_inv[head] * edge_weight * degree_inv[tail]
        return norm
    def scatter_function(self,type_, src, index, dim_size = None):
        '''
        This function given scatter_ type, which should me max, mean,or sum, given source array, given index array, given dimension size
        '''
        assert type_.lower() in ["sum","mean","max"]
        return torch_scatter.scatter(src, index, dim=0,out=None,dim_size = dim_size, reduce= type_)
    
    def propogating_message(self, method, node_features,edge_index,edge_type, rel_embedding, edge_norm,mode,type_):
        '''
        This function done the basic aggregation
        '''
        assert method in ["sum", "mean", "max"]
        assert mode in ["in","out","loop"]
        size = node_features.shape[0]
        coresponding_weight = getattr(self, 'w_{}'.format(mode))
        #-------------- this index selection: given relation embedding and relation_basic representation, choose the inital basis vector part
        relation_embedding = torch.index_select(rel_embedding,dim = 0, index = edge_type)
        # ------------- using index of tail in edge index to represent head by relation
        node_features = node_features[edge_index[1]]
        out = self.relation_transform(node_features, relation_embedding,type_)
        out = torch.matmul(out,coresponding_weight)
        out = out if edge_norm is None else out * edge_norm.view(-1, 1)
        out = self.scatter_function(method,out,edge_index[0],  size)
        return out    
    def forward(self, nodes_features, edge_index,edge_type):
        '''
        Forward propogate function:
            Given input nodes_features, adj_matrix, relation_matrix
        '''
        with amp.autocast():
            if self.device is None:
                self.device = edge_index.device
            # ----------- First done the basis part, which means represent each relation using a vector space defining previously
            relation_embedding = torch.mm(self.rel_weight,self.basis_vector)
            # ----------- add a self-loop dimension
            relation_embedding = torch.cat([relation_embedding,self.loop_rel],dim = 0)
            # print(edge_index.shape)
            num_edges = edge_index.shape[1]//2
            num_nodes = nodes_features.shape[self.nodes_dim]
            if not self.cache or self.in_norm == None:
                #---------------- in represent in_relation, out represent out_relation
                self.in_index, self.out_index = edge_index[:,:num_edges], edge_index[:,num_edges:]
                self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]
                # --------------- create self-loop part
                self.loop_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]).to(self.device)
                self.loop_type = torch.full((num_nodes,), relation_embedding.shape[0]-1, dtype = torch.long).to(self.device)
                # -------------- create normalization part
                self.in_norm = self.normalization(self.in_index, num_nodes)
                self.out_norm = self.normalization(self.out_index, num_nodes)
            #print(self.in_norm.isinf().any())
            in_res = self.propogating_message('sum',nodes_features,self.in_index,self.in_type, relation_embedding,self.in_norm,"in",self.comp_type)
            loop_res = self.propogating_message('sum',nodes_features,self.loop_index,self.loop_type, relation_embedding,None,"loop",self.comp_type)
            out_res = self.propogating_message('sum',nodes_features,self.out_index,self.out_type, relation_embedding,self.out_norm,"out",self.comp_type)
            # I don't know why but source code done it
            out = self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)
            # update the relation embedding
            out_2 = torch.matmul(relation_embedding,self.weight_rel)
            return self.act(out),out_2
            
            

class CompGcn_non_first_layer(nn.Module):
    nodes_dim = 0
    head_dim = 0
    tail_dim = 1
    def __init__(self, in_channels, out_channels, num_relations,act = torch.tanh,dropout = 0.2, comp_type = "sub"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.act = act
        self.device = None
        self.comp_type = comp_type
        # this learnable weight matrix is for projection, that project each relation to the same dimension of node_dimension
        self.weight_rel = get_param((in_channels,out_channels))
        # add another embedding for loop
        self.loop_rel = get_param((1,in_channels))
        #----------- Creating three updated matrix, as three kind of relations updating, in, out, loop
        # using for updating weight
        self.w_in = get_param((in_channels,out_channels))
        self.w_out = get_param((in_channels,out_channels))
        self.w_loop = get_param((in_channels,out_channels))
        self.drop = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_channels)
    def relation_transform(self, entity_embedding, relation_embedding,type_):
        '''
        This function given entity embedding and relation embedding, in order return three types of 
        non-parameterized operations, which is subjection, corr, multiplication
        '''
        assert type_ in ["mul","sub","corr"], "not implemented now"
        if type_ == "mul":
            out = entity_embedding*relation_embedding
        elif type_ == "sub":
            out = entity_embedding - relation_embedding
        else:
            out = ccorr(entity_embedding,relation_embedding)
        return out
    
    def normalization(self, edge_index, num_entity):
        '''
        As normal GCN, this function calculate the normalization adj matrix 
        '''
        head, tail = edge_index
        edge_weight = torch.ones_like(head).float()
        degree = torch_scatter.scatter_add(edge_weight,head,dim_size=num_entity,dim = self.nodes_dim)
        degree_inv = degree.pow(-0.5)
        # if inf, in order to prevent nan in scatter function
        degree_inv[degree_inv == float("inf")] = 0
        norm = degree_inv[head] * edge_weight * degree_inv[tail]
        return norm
    def scatter_function(self,type_, src, index, dim_size = None):
        '''
        This function given scatter_ type, which should me max, mean,or sum, given source array, given index array, given dimension size
        '''
        assert type_.lower() in ["sum","mean","max"]
        return torch_scatter.scatter(src, index, dim=0,out=None,dim_size = dim_size, reduce= type_)
    
    def propogating_message(self, method, node_features,edge_index,edge_type, rel_embedding, edge_norm,mode,type_):
        '''
        This function done the basic aggregation
        '''
        assert method in ["sum", "mean", "max"]
        assert mode in ["in","out","loop"]
        size = node_features.shape[0]
        coresponding_weight = getattr(self, 'w_{}'.format(mode))
        #-------------- this index selection: given relation embedding and relation_basic representation, choose the inital basis vector part
        # print(edge_type)
        relation_embedding = torch.index_select(rel_embedding,dim = 0, index = edge_type)
        # ------------- using index of tail in edge index to represent head by relation
        node_features = node_features[edge_index[1]]
        out = self.relation_transform(node_features, relation_embedding,type_)
        out = torch.matmul(out,coresponding_weight)
        out = out if edge_norm is None else out * edge_norm.view(-1, 1)
        out = self.scatter_function(method,out,edge_index[0],  size)
        return out    
    def forward(self, nodes_features, edge_index,edge_type,relation_embedding):
        '''
        Forward propogate function:
            Given input nodes_features, adj_matrix, relation_matrix
        '''
        with amp.autocast():
            if self.device is None:
                self.device = edge_index.device
            # ----------- add a self-loop dimension
            relation_embedding = torch.cat([relation_embedding,self.loop_rel],dim = 0)
            # print(edge_index.shape)
            num_edges = edge_index.shape[1]//2
            num_nodes = nodes_features.shape[self.nodes_dim]
            #---------------- in represent in_relation, out represent out_relation
            self.in_index, self.out_index = edge_index[:,:num_edges], edge_index[:,num_edges:]
            self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]
            # --------------- create self-loop part
            self.loop_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]).to(self.device)
            self.loop_type = torch.full((num_nodes,), relation_embedding.shape[0]-1, dtype = torch.long).to(self.device)
            # -------------- create normalization part
            self.in_norm = self.normalization(self.in_index, num_nodes)
            self.out_norm = self.normalization(self.out_index, num_nodes)
            #print(self.in_norm.isinf().any())
            in_res = self.propogating_message('sum',nodes_features,self.in_index,self.in_type, relation_embedding,self.in_norm,"in",self.comp_type)
            loop_res = self.propogating_message('sum',nodes_features,self.loop_index,self.loop_type, relation_embedding,None,"loop",self.comp_type)
            out_res = self.propogating_message('sum',nodes_features,self.out_index,self.out_type, relation_embedding,self.out_norm,"out",self.comp_type)
            # I don't know why but source code done it
            out = self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)
            # update the relation embedding
            out_2 = torch.matmul(relation_embedding,self.weight_rel)
            return self.act(out),out_2[:-1]# ignoring self loop inserted 

class CompGcn_total(nn.Module):
    def __init__(self, channel_ls ,num_relation, num_basis_vector, num_layers = 2, basis = False, comp_type = "sub"):
        '''
        Notice that in preprocessing, we assume that the node number will not be changed on the graph, only change relation.
        input params:
            1. channel_ls: a channel list containing all conv channel
            2. num_relation, the number of relations_type for each graph(after preprocessing, should be the same for each graph)
            3. num_basis_vector, the first layer basis of first graph.
            4. basis, whether need basis
        '''
        assert len(channel_ls) == num_layers + 1 , "channel number should be layer numbers + 1 , got length "+str(len(channel_ls))+" with number of layers "+str(num_layers)
        super(CompGcn_total, self).__init__()
        self.basis = basis
        self.GCN_block = nn.Sequential()
        for i in range(num_layers):
            if basis and i == 0:
                self.GCN_block.add_module("Basis_conv_layer",  CompGcnBasis(in_channels = channel_ls[0], out_channels= channel_ls[1],
                                                                            num_relations=num_relation,
                                                                            num_basis_vector= num_basis_vector, comp_type = comp_type)) 
            else:
                self.GCN_block.add_module("Conv_layer"+str(i),CompGcn_non_first_layer(channel_ls[i], channel_ls[i+1], num_relation, comp_type = comp_type))
    def forward(self, init_features = None, node_embd = None,rel_embd = None,device = None, edge_idx = None, edge_type = None):
        with amp.autocast():
            for i, blk in enumerate(self.GCN_block):
                if self.basis and i == 0:
                    node_embd, rel_embd = blk(init_features, edge_idx.to(device), edge_type.to(device))
                else:
                    node_embd, rel_embd = blk(node_embd, edge_idx.to(device), edge_type.to(device), rel_embd)
            return node_embd, rel_embd

