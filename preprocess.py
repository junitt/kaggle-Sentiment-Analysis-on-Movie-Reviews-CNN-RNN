import pandas as pd 
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize
from util import *
train_data=pd.read_csv('./data/train.tsv.zip',sep='\t')
test_data=pd.read_csv('./data/test.tsv.zip',sep='\t')
train_data=train_data.drop(['PhraseId','SentenceId'],axis=1)
test_data=test_data.drop(['PhraseId','SentenceId'],axis=1)
all_doc = pd.concat([train_data,test_data],axis = 0,ignore_index = True)
phrases = all_doc['Phrase'].tolist()
lst=[]
for sent in tqdm(phrases):
    lst.extend(getwords(sent))
s=set(lst)
with open('./data/samr.vocab', 'w') as f:
    for word in tqdm(s):
        f.writelines(word+'\n')
    f.close()    