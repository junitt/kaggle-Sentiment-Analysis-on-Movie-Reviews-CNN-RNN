import torch
import torch.nn as nn
import os
from torch.utils import data
from tqdm import tqdm
from cnnnet import CNN,myRnn
import samr
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

writer = SummaryWriter()
lr = 0.01#cnn 1e-2 or 5e-3
batch = 1024
epochs = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_dim = 50
filter_sizes=[1,2,3,4,5]
n_filters = 100
dropout=0.5
num_classes = 5#情感的种类总数

hidden_dim=64
num_layers=2
modelname='rnnmodel'
datasettype='glove'

def train_epoch(model, dataloader, criterion: dict, optimizer,
                scheduler, epoch, device):
    model.train()
    bar = tqdm(dataloader)
    bar.set_description(f'epoch {epoch:2}')
    correct, total = 0, 0
    last_loss=0
    for dat in bar:
        text,label,l=dat
        text=text.to(device)
        label=label.to(device)
        if modelname=='cnnmodel':
            predicted_label = model(text)
        else:
            predicted_label = model(text,l)
        optimizer.zero_grad()
        loss = criterion(predicted_label, label)
        loss.backward()
        optimizer.step()
        total+=len(label)
        correct+=(predicted_label.argmax(1) == label).sum().item()

        bar.set_postfix_str(f'lr={scheduler.get_last_lr()[0]:.4f} acc={correct / total * 100:.2f} loss={loss.item():.2f}')
        last_loss=loss.item()
        scheduler.step()
    writer.add_scalar('Accuracy/train',correct / total * 100, epoch)
    writer.add_scalar('Loss/train',last_loss, epoch)


def test_epoch(model, dataloader, device, epoch):
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for dat in dataloader:
            text,label,l=dat
            text=text.to(device)
            label=label.to(device)
            if modelname=='cnnmodel':
                predicted_label = model(text)
            else:
                predicted_label = model(text,l)
            correct+=(predicted_label.argmax(1) == label).sum().item()
            total+=len(label)

        print(f' val acc: {correct / total * 100:.2f}')
        writer.add_scalar('Accuracy/test',correct / total * 100, epoch)
        return correct / total

def main():
    #model=myRnn(vocab_size=400,embed_dim=embed_dim,hidden_dim=hidden_dim,num_layers=num_layers, dropout=dropout,pad_idx=5,ext_info=None)
    root='./data'
    workspace_dir='./models'
    trainset=samr.get_dataset(root,mode='train',datasettype=datasettype,embedding_dim=embed_dim)
    trainloader = data.DataLoader(trainset,#修改了下路径
                                  batch_size=batch, shuffle=True)
    testloader = data.DataLoader(samr.get_dataset(root,mode='valid',datasettype=datasettype,embedding_dim=embed_dim),
                                 batch_size=batch, shuffle=False)
    vocab_size=len(trainset.vocab)
    ext_info=None
    if datasettype=='glove':
        ext_info={'glo':trainset.glove,'ori_len':trainset.vocab_len}
        vocab_size=trainset.vocab_len
    if modelname=='cnnmodel':
        model = CNN(vocab_size=vocab_size,embed_dim=embed_dim,n_filters=n_filters, filter_sizes=filter_sizes, num_classes=num_classes, 
                dropout=dropout, pad_idx=trainset.pad_idx,ext_info=ext_info).to(device)
    else:
        model=myRnn(vocab_size=vocab_size,num_class=num_classes,embed_dim=embed_dim,hidden_dim=hidden_dim,
                    num_layers=num_layers, dropout=dropout,pad_idx=trainset.pad_idx,ext_info=ext_info).to(device)
    optimizer=torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    #optimizer=AdamW(model.parameters(), lr=lr)
    t_total=len(trainloader)*epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=t_total)
    criterion =nn.CrossEntropyLoss()
    ma=0
    for epoch in range(epochs):
        train_epoch(model, trainloader, criterion, optimizer,
                    scheduler, epoch, device)
        precesion=test_epoch(model, testloader, device, epoch)#precesion 验证集的准确率
        if precesion>ma:
            if modelname=='cnnmodel':
                torch.save(model.state_dict(), os.path.join(workspace_dir, f'{modelname}_{datasettype}_{embed_dim}_{n_filters}_g.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(workspace_dir, f'{modelname}_{datasettype}_{embed_dim}_{hidden_dim}_{num_layers}_g.pth'))
            ma=precesion#ma是之前epoch最大的准确率
    writer.flush()
    


if __name__ == '__main__':
    main()