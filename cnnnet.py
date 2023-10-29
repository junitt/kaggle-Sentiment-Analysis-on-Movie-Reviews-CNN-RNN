import torch
from torch import nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes, num_classes, 
                 dropout, pad_idx,ext_info=None):
        super().__init__() 
        self.embedding=nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if ext_info is not None:
            pad=torch.zeros((1,embed_dim))
            extend_word=torch.randn((vocab_size-pad_idx-1,embed_dim))
            self.embedding.weight.data.copy_(torch.concat((ext_info['glo'].vectors,pad,extend_word)))
        self.convs=nn.ModuleList([
                                    nn.Conv2d(in_channels=1, 
                                              out_channels=n_filters, 
                                              kernel_size=(fs, embed_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        #self.fc=nn.Linear(len(filter_sizes) * n_filters, num_classes)
        self.linrnn=nn.Sequential(
            nn.Linear(len(filter_sizes) * n_filters, 40),
            nn.ReLU(),
            nn.Linear(40, num_classes)
        )
        self.dropout=nn.Dropout(dropout)
        
    def forward(self, text):
        embedded=self.embedding(text)
        embedded=embedded.unsqueeze(1)
        conved=[F.relu(conv(embedded)).squeeze(3) for conv in self.convs]    
        pooled=[F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat=self.dropout(torch.cat(pooled, dim=1))
        return self.linrnn(cat)
    
class myRnn(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class,hidden_dim,num_layers, dropout,pad_idx,ext_info=None):
        super().__init__() 
        self.embedding=nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.hidden_dim=hidden_dim
        if ext_info is not None:
            pad=torch.zeros((1,embed_dim))
            extend_word=torch.randn((vocab_size-pad_idx-1,embed_dim))
            self.embedding.weight.data.copy_(torch.concat((ext_info['glo'].vectors,pad,extend_word)))
        self.linrnn=nn.Sequential(
            nn.Linear(self.hidden_dim*2, 8),
            nn.ReLU(),
            nn.Linear(8, num_class)
        )
        self.dropout=nn.Dropout(dropout)
        self.rnn=nn.RNN(embed_dim,hidden_dim,num_layers,nonlinearity='relu',batch_first=True,dropout=0,bidirectional=True)

    def forward(self,text,lenlist):
        embedded=self.embedding(text)#设置paddingidx为vocab_size-1
        packedseq=torch.nn.utils.rnn.pack_padded_sequence(embedded,torch.Tensor(lenlist),batch_first=True,enforce_sorted=False)
        #packedseq=packedseq.to(self.device)
        output,h_n=self.rnn(packedseq)
        hidden=self.dropout(torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1))
        return self.linrnn(hidden.squeeze(0))