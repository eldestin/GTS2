from utils import *
import torch
from torch import nn

class language_Dataset(torch.utils.data.Dataset):
    def __init__(self, df, edge_idx, edge_type):
        '''
        df is dataframe given previously
        '''
        self.df = df.reset_index(drop = True)
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.edge_idx = edge_idx
        self.edge_type = edge_type
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        '''
        This function will return the index
        '''
        dic = change_input(self.tokenizer,
                self.df.loc[idx, "head"],
                self.df.loc[idx, "relation"], 
                self.df.loc[idx,"tail"],
                self.df.loc[idx,"labels"])
        return torch.tensor(dic["input_ids"]), torch.tensor(dic["segment_ids"]), torch.tensor(dic["attention_mask"]), torch.tensor(dic["labels"]), self.df.loc[idx,"index_where"], torch.tensor(dic["head_idx"]),torch.tensor(dic["relation_idx"]), torch.tensor(dic["tail_idx"])
