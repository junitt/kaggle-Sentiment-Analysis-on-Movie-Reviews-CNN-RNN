import re
from nltk.tokenize import word_tokenize
def getwords(sent):
    sent=' '+sent
    sent+=' '
    sent=re.sub(r' [\d]* ', ' __mynum__ ', str(sent).lower())
    words = word_tokenize(sent)
    return words

def decode(keys:list,lst):
    ret=''
    for x in lst:
        ret+=keys[x.item()]
        ret+=' '
    return ret

def my_tokenize(vocab,words):
    lst=[]
    for word in words:
        lst.append(vocab[word])
    return lst