from tqdm import tqdm
from samr import *
import numpy as np
from cnnnet import CNN,myRnn
from torch.utils.data import DataLoader
root='./data'
df_test = pd.read_csv('./data/test.tsv.zip', sep='\t')

test_id=df_test["PhraseId"]
embed_dim = 50
filter_sizes=[1,2,3,4,5]
n_filters = 100
dropout=0.5
num_classes = 5#情感的种类总数
datasettype='glove'
modelname='rnnmodel'

dataset=get_dataset(root,mode='test',datasettype=datasettype,embedding_dim=embed_dim)
testloader = DataLoader(dataset,
            batch_size=512, shuffle=False)

predict = np.array([])
predict = torch.tensor(predict)
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size=len(dataset.vocab)
hidden_dim=64
num_layers=2
ext_info=None
if datasettype=='glove':
    ext_info={'glo':dataset.glove,'ori_len':dataset.vocab_len}
if modelname=='cnnmodel':
    model = CNN(vocab_size=vocab_size,embed_dim=embed_dim,n_filters=n_filters, filter_sizes=filter_sizes, num_classes=num_classes, 
                dropout=dropout, pad_idx=dataset.pad_idx,ext_info=ext_info).to(device)
else:
    model=myRnn(vocab_size=vocab_size,num_class=num_classes,embed_dim=embed_dim,hidden_dim=hidden_dim,
                    num_layers=num_layers, dropout=dropout,pad_idx=dataset.pad_idx,ext_info=ext_info).to(device)
if modelname=='cnnmodel':
    model.load_state_dict(torch.load(f"./models/{modelname}_{datasettype}_{embed_dim}_{n_filters}_g.pth"))
else:
    model.load_state_dict(torch.load(f"./models/{modelname}_{datasettype}_{embed_dim}_{hidden_dim}_{num_layers}_g.pth"))
model.eval()
with torch.no_grad():
    for dat in tqdm(testloader):
        if modelname=='rnnmodel':
            text,l=dat
            text=text.to(device)
            predicted_label = model(text,l)
        else:
            text,l=dat
            text=text.to(device)
            predicted_label = model(text)
        predicted_label=predicted_label.argmax(1).cpu()
        predicted_label=predicted_label.view(-1,)
        predict = torch.cat((predict, predicted_label), dim=0)
predict=predict.type(torch.int64)
predict=predict.view(-1,1).numpy().squeeze(1)

submit_df = pd.DataFrame({"PhraseId":test_id,
                         "Sentiment":predict})
submit_df.to_csv(f"./data/{modelname[:3]}{datasettype}submit.csv", index=False)