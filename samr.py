import os
from tensorflow import keras
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from util import *
import torchtext

class mydataset(Dataset):
    def __init__(self,root,mode,embedding_dim,vocab_fileordir):
        self.path=root
        self.mode=mode
        self.embedding_dim=embedding_dim
        self.vocab_source=vocab_fileordir
    
    def get_tokenized_data(self):
        data=pd.read_csv(self.path,sep='\t')
        data=data.drop(['PhraseId','SentenceId'],axis=1)
        all_sents=data['Phrase'].tolist()
        self.labels=[]
        all_labels=None
        if self.mode!='test':
            all_labels=data['Sentiment'].tolist()
        if self.mode=='train':
            self.sents=[]
            lst=self.getvalidid()
            pt=0
            for idx,sent in enumerate(all_sents):
                if pt==len(lst):
                    self.sents.append(sent)
                    self.labels.append(all_labels[idx])
                else:
                    if idx==lst[pt]:
                        pt+=1
                    else:
                        self.sents.append(sent)
                        self.labels.append(all_labels[idx])
        elif self.mode=='valid':
            self.sents=[]
            lst=self.getvalidid()
            for x in lst:
                self.sents.append(all_sents[x])
                self.labels.append(all_labels[x])
        else:
            self.sents=all_sents
        self.tokenized_sent=[my_tokenize(vocab=self.vocab,words=getwords(sent)) for sent in self.sents]
    def getvalidid(self):
        data=pd.read_csv('./data/validnum.csv',sep='\t')
        lst=[int(s) for s in data['Sentiment'].tolist()]
        return lst

class samrdataset(mydataset):
    def __init__(self,root,mode,embedding_dim,vocab_fileordir):
        super(samrdataset, self).__init__(root,mode,embedding_dim,vocab_fileordir)
        vocab=dict()
        with open(vocab_fileordir,'r') as f:
            i=0
            for line in f:
                vocab[line.split('\n')[0]]=i
                i+=1
        self.vocab=vocab
        self.pad_idx=i
        self.vocab['<pad>']=self.pad_idx
        self.keys=list(vocab.keys())
        super().get_tokenized_data()
        self.datalen=len(self.sents)

    def __getitem__(self, index):
        assert len(self.tokenized_sent[index])!=0
        l=len(self.tokenized_sent[index])
        self.tokenized_sent[index]=keras.preprocessing.sequence.pad_sequences([self.tokenized_sent[index]],maxlen=56,padding='post',value=self.pad_idx,truncating='post')[0]
        if self.mode!='test':
            return self.tokenized_sent[index],int(self.labels[index]),l
        else:
            return self.tokenized_sent[index],l

    def __len__(self):
        return self.datalen
    
class gloveSamrDataset(mydataset):
    def __init__(self,root,mode,embedding_dim,vocab_fileordir):
        super(gloveSamrDataset, self).__init__(root,mode,embedding_dim,vocab_fileordir)
        self.glove=torchtext.vocab.GloVe(name='6B', dim=embedding_dim, cache=vocab_fileordir) 
        self.vocab=self.glove.stoi
        self.pad_idx=len(self.vocab)#也是原来的字典大小
        now_max=self.pad_idx
        self.vocab['<pad>']=self.pad_idx
        with open('./data/samr.vocab','r') as f:
            now_max+=1
            for line in f:
                word=line.split('\n')[0]
                if word not in self.vocab:
                    self.vocab[word]=now_max
                    now_max+=1
        self.keys=list(self.vocab.keys())
        self.vocab_len=len(self.keys)
        super().get_tokenized_data()
        self.datalen=len(self.sents)

    def __getitem__(self, index):
        assert len(self.tokenized_sent[index])!=0
        l=len(self.tokenized_sent[index])
        self.tokenized_sent[index]=keras.preprocessing.sequence.pad_sequences([self.tokenized_sent[index]],maxlen=56,padding='post',value=self.pad_idx,truncating='post')[0]
        if self.mode!='test':
            return self.tokenized_sent[index],int(self.labels[index]),l
        else:
            return self.tokenized_sent[index],l

    def __len__(self):
        return self.datalen

def get_dataset(root,mode,datasettype,embedding_dim):
    if mode!='test':
        cmt_pth=os.path.join(root,'train.tsv.zip')
    else:
        cmt_pth=os.path.join(root,'test.tsv.zip')
    if datasettype=="random":
        vocab_source='./data/samr.vocab'
        return samrdataset(cmt_pth,mode,embedding_dim,vocab_source)
    elif datasettype=='glove':
        vocab_source='./glove_cache'
        return gloveSamrDataset(cmt_pth,mode,embedding_dim,vocab_source)


if __name__ == '__main__':
    root='./data'
    train_data=get_dataset(root,'valid',datasettype='random',embedding_dim=200)
    train_loader = torch.utils.data.DataLoader(train_data, 
        batch_size=1024, # 每批样本个数
        shuffle=True, # 是否打乱顺序
    )
    for data,lab,ll in train_loader:
        print(decode(train_data.keys,data[0]))
        print('')