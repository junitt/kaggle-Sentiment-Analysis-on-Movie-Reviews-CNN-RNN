import random
import pandas as pd
Sample_cnt=1000
lst=[[]for _ in range(5)]
train_data=pd.read_csv('./data/train.tsv.zip',sep='\t')
train_data=train_data.drop(['PhraseId','SentenceId'],axis=1)
labels=train_data['Sentiment'].tolist()
fin=[]
for idx,lab in enumerate(labels):
    lst[lab].append(idx)
for i in range(5):
    temp=[]
    if i!=2:
        temp=random.sample(range(0, len(lst[i])), Sample_cnt//2//4)
    else:
        temp=random.sample(range(0, len(lst[i])), Sample_cnt//2)
    for u in temp:
        fin.append(lst[i][u])
fin=sorted(fin)
submit_df = pd.DataFrame({
                         "Sentiment":fin})
submit_df.to_csv("./data/validnum.csv", index=False)