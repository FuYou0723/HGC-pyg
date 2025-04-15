
from torch_geometric.data import InMemoryDataset, DataLoader
from datasets import CustomDataset
from model import myHGNN
import numpy as np 
import torch
device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
dataset=CustomDataset(root='/home/zhangyifan/alive/HC-pyg',\
    data_type='hypergraph', folder='RHG', name='RHG_3')
import torch
torch.manual_seed(12345)
data_=dataset.shuffle()

# 80% for train and 20% for test
idx_train=int(0.8*len(data_))
train_dataset=data_[:idx_train]
test_dataset=data_[idx_train:]
print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
from torch_geometric.data import DataLoader
train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=32, shuffle=False)
for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()
times=[]
from model import myHGNN
in_ft=data_.num_features
out_ft=data_.num_classes
hid_ft=128
for i in range(11):
    print(f'run {i}')
    model=myHGNN(in_ft,out_ft,hid_ft,num_layers=3,dropout=0.5)
    model=model.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=5e-4)
    cir=torch.nn.CrossEntropyLoss()
    import pdb 
    #pdb.set_trace()
    def train():
        model.train()
        t_loss=0.0
        num_batches=len(train_loader)
        for data in train_loader:
            opt.zero_grad()
            #pdb.set_trace()
            out=model(data.to(device))
            loss=cir(out,data.y)
            loss.backward()
            opt.step()
            t_loss+=loss.item()
        mean_loss=t_loss/num_batches
        return mean_loss
    def test():
        model.eval()
        corr=0.0
        for data in test_loader:
            out=model(data.to(device=device))
            pred=out.argmax(dim=1)
            # print(pred.shape)
            # print(data.y.shape)
            corr+=(int((pred==data.y.to(device)).sum()))
        return corr/len(test_dataset)
    losses=[]   
    test_accs=[]
    n_epochs=170
    epochs=range(1,n_epochs+1)
    for epoch in epochs:
        loss=train()
        #losses.append(loss)
        test_acc=test()
        print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}')
        test_accs.append(test_acc)
# 计算均值和方差
mean_test_acc = np.mean(test_accs)
var_test_acc = np.var(test_accs)

print(f'10 次运行测试准确率的均值: {mean_test_acc:.4f}')
print(f'10 次运行测试准确率的方差: {var_test_acc:.4f}')