import os 
os.environ['CUDA_VISIBLE_DEVICES']='1'

import numpy as np
import networkx as nx
import torch
import random
import torch.backends.cudnn as cudnn
seed=0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import knn_graph,GATConv
from torch_geometric.utils import (negative_sampling, remove_self_loops,to_undirected,
                                   add_self_loops)
from tqdm import trange
from dataset import *
from layer import DHGLayer
from sklearn.metrics import f1_score,accuracy_score

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

def recon_loss(product, z, pos_edge_index, neg_edge_index_0=None):
        EPS = 1e-15
        pos_loss = -torch.log(
            product(z, pos_edge_index, sigmoid=True) + EPS).mean()

        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        if neg_edge_index_0 is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        else :
            neg_edge_index = neg_edge_index_0

        neg_loss = -torch.log(1 -product(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss
    
def recon_loss1(product, z, pos_edge_index,neg_edge_index):
        EPS = 1e-15
        pos_loss = -torch.log(
            product(z, pos_edge_index, sigmoid=True) + EPS).mean()

        neg_loss = -torch.log(1 -
                              product(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()
        return pos_loss + neg_loss

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.net1=nn.Linear(100,50)
        self.net2=nn.Linear(50,1)
        self.BN=nn.BatchNorm1d(50)
        self.reset_parameters()
    
    def reset_parameters(self):
        init.normal(self.net1.weight)
        init.normal(self.net2.weight)
        init.constant_(self.net1.bias, 0)
        init.constant_(self.net2.bias, 0)
        init.constant(self.BN.weight, 1)
        init.constant(self.BN.bias, 0)

    def forward(self, x):
        x=self.net1(x)
        x=self.BN(x)
        x=F.sigmoid(x)
        x=self.net2(x)
        return x

class Encoder(torch.nn.Module):
    def __init__(self,res=True):
        super(Encoder, self).__init__()
        #文本
        self.feature_conv1 = GATConv(200, 100, dropout=0.6, negative_slope=0.2, heads=2, concat=False)
        self.feature_conv2 = DHGLayer(dim_in=100,dim_out=100,activation=nn.ReLU(),
                                structured_neighbor=25,
                                nearest_neighbor=25,
                                cluster_neighbor=25,
                                wu_knn=1,
                                wu_kmeans=2,
                                wu_struct=2,
                                n_cluster=100,
                                n_center=1,
                                has_bias=True)
        
        self.feature_conv3 = GATConv(100, 100, dropout=0.6, negative_slope=0.2, heads=2, concat=False)
        #结构
        self.conv1 = GATConv(200, 100, dropout=0.6, negative_slope=0.2, heads=2, concat=False)
        self.conv2 = DHGLayer(dim_in=100,dim_out=100,activation=nn.ReLU(),
                                structured_neighbor=25,
                                nearest_neighbor=25,
                                cluster_neighbor=25,
                                wu_knn=1,
                                wu_kmeans=2,
                                wu_struct=2,
                                n_cluster=100,
                                n_center=1,
                                has_bias=True)
        
        self.conv3 = GATConv(100, 100, dropout=0.6, negative_slope=0.2, heads=2, concat=False)
       
        self.BN=nn.BatchNorm1d(100)
        self.BN1=nn.BatchNorm1d(100)
        init.constant(self.BN.weight, 1)
        init.constant(self.BN.bias, 0)
        init.constant(self.BN1.weight, 1)
        init.constant(self.BN1.bias, 0)
        
        self.atten1=Attention()
        self.atten2=Attention()
        

    def forward(self, inputx, edge_index, fea_edge_index,edge_list):

        #_,feax1 = self.feature_conv1( inputx, fea_edge_index)
        feax1 = self.feature_conv1( inputx, fea_edge_index)
        feax1=self.BN(feax1)
        feax1=F.relu(feax1)
        
        #_,x1 = self.conv1(inputx, edge_index)
        x1 = self.conv1(inputx, edge_index)
        x1=self.BN1(x1)
        x1=F.relu(x1)
        
        feax1 = self.feature_conv2( feax1, edge_list, 1)
        #_,feax2 = self.feature_conv3( feax1, fea_edge_index)
        feax2 = self.feature_conv3( feax1, fea_edge_index)
        
        x1=self.conv2( x1, edge_list, 1)
        #_,x2 = self.conv3(x1 , edge_index)
        x2 = self.conv3(x1 , edge_index)
        
        att1=self.atten1(feax2)
        att2=self.atten2(x2)
        att1,att2 = torch.split(F.softmax(torch.cat((att1,att2),1),dim=1),1,dim=1)
        output=att1*feax2+att2*x2
       
        return output

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.net1=nn.Linear(200,100)
        self.net2=nn.Linear(100,2)
        self.BN=nn.BatchNorm1d(100)
        self.reset_parameters()
    
    def reset_parameters(self):
        init.normal(self.net1.weight)
        init.normal(self.net2.weight)
        init.constant_(self.net1.bias, 0)
        init.constant_(self.net2.bias, 0)
        init.constant(self.BN.weight, 1)
        init.constant(self.BN.bias, 0)

    def forward(self, x):
        x=self.net1(x)
        x=self.BN(x)
        x=F.relu(x)
        probability=self.net2(x)
        predictions = torch.argmax(probability, dim=1)
        return probability,predictions

def normalize_features(x):
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
    x = x / x_norm
    return x

###########################dataprocess##############################
print('start reading')
##############读取文本嵌入############
sentences = []
with open('../doc2vec/doc2vec_tmp/docvecs.txt','r') as fileTrainRaw:
    d=fileTrainRaw.read()
    sentences=eval(d)
douban=sentences[0:4399]
weibo=sentences[4399:]
douban=normalize_features(douban)
weibo=normalize_features(weibo)
douban=Tensor(douban)
weibo=Tensor(weibo)
############读取边缘信息###############
doubanedge=[]
with open('../dataset/loop-all-doubanedge.txt',"r") as f:
    doubanedge=eval(f.read())
doubanpiars=[[x[0] for x in doubanedge],[x[1] for x in doubanedge]]
doubanedges = torch.LongTensor(doubanpiars).cuda()
weiboedge=[]
with open('../dataset/loop-all-weiboedge.txt',"r") as f:
    weiboedge=eval(f.read())
weibopiars=[[x[0] for x in weiboedge],[x[1] for x in weiboedge]]
weiboedges = torch.LongTensor(weibopiars).cuda()
print(doubanedges.shape,weiboedges.shape)
#######采用knn的方法
doubanfeatureedges = knn_graph(douban, k=27, loop=True)
weibofeatureedges = knn_graph(weibo, k=27, loop=True)
doubanfeatureedges = to_undirected(doubanfeatureedges)
weibofeatureedges = to_undirected(weibofeatureedges)
print(doubanfeatureedges.shape,weibofeatureedges.shape)
#############构造的图信息###################
douban_G=nx.from_edgelist(doubanedge)
douban_edge_dict = douban_G.adjacency_list()
weibo_G=nx.from_edgelist(weiboedge)
weibo_edge_dict = weibo_G.adjacency_list()
########################################
doubanedge1=[]
with open('../doc2vec/doubansimpiar.txt',"r") as f:
    doubanedge1=eval(f.read())
doubanpiars1=[[x[0] for x in doubanedge1],[x[1] for x in doubanedge1]]
doubanedges1 = torch.LongTensor(doubanpiars1).cuda()
weiboedge1=[]
with open('../doc2vec/weibosimpiar.txt',"r") as f:
    weiboedge1=eval(f.read())
weibopiars1=[[x[0] for x in weiboedge1],[x[1] for x in weiboedge1]]
weiboedges1 = torch.LongTensor(weibopiars1).cuda()
########################################
doubanedge2=[]
with open('../doc2vec/doubannegsimpiar.txt',"r") as f:
    doubanedge2=eval(f.read())
doubanpiars2=[[x[0] for x in doubanedge2],[x[1] for x in doubanedge2]]
doubanedges2 = torch.LongTensor(doubanpiars2).cuda()
weiboedge2=[]
with open('../doc2vec/weibonegsimpiar.txt',"r") as f:
    weiboedge2=eval(f.read())
weibopiars2=[[x[0] for x in weiboedge2],[x[1] for x in weiboedge2]]
weiboedges2 = torch.LongTensor(weibopiars2).cuda()
#################读取节点对#################
data=Getdata()
valdata,valdatalabel=data.getval()
testdata,testdatalabel=data.gettest()
testdata=testdata+valdata
testdatalabel=testdatalabel+valdatalabel
print(len(valdata),len(testdata))

doubanencoder= Encoder().cuda()
weiboencoder=Encoder().cuda()
gnnnet=AutoEncoder().cuda()
DEnoptimizer = torch.optim.Adam(doubanencoder.parameters(), lr=0.0003)
WEnoptimizer = torch.optim.Adam(weiboencoder.parameters(), lr=0.0003)
gnnnetoptimizer = torch.optim.Adam(gnnnet.parameters(), lr=0.0005)

product=InnerProductDecoder()
loss_fn = nn.MSELoss(reduce = True,size_average = True)
cro_loss = nn.CrossEntropyLoss(reduce = True,size_average = True)
softmax = nn.Softmax(dim=1)
###########################training###################################
print('start training')

bestepoch=0
bestacc=0
bestf1=0
bestloss=100
epochs=50000
tag='-pygdhgnn-50000-0.05'

for epoch in trange(epochs):
    if epoch % 50 == 0:
        traindata,trainlabel=data.gettrain()
        batch_x=Batch(traindata,128,len(traindata))
        batch_label=Batch(trainlabel,128,len(trainlabel))

    doubanencoder.train()
    weiboencoder.train()
    gnnnet.train()
        
    piars=batch_x.getbatch()
    label=batch_label.getbatch()
    x=[piar[0] for piar in piars]
    y=[piar[1] for piar in piars]
    
    h1 = doubanencoder(douban, doubanedges, doubanfeatureedges, douban_edge_dict)
    h2 = weiboencoder(weibo, weiboedges, weibofeatureedges, weibo_edge_dict)
    
    #分类loss
    doubansample=h1[x]
    weibosample=h2[y]
    pro,_=gnnnet(torch.cat((doubansample,weibosample),1))
    classfi_loss = cro_loss(F.softmax(pro,dim=1),torch.tensor(label).cuda())
    
    #构造loss
    doubanrecloss=recon_loss(product,h1,doubanedges)
    weiborecloss=recon_loss(product,h2,weiboedges)
    doubanrecloss1=recon_loss1(product,h1,doubanedges1,doubanedges2)
    weiborecloss1=recon_loss1(product,h2,weiboedges1,weiboedges2)

    allloss=classfi_loss+0.2*doubanrecloss+0.2*weiborecloss+0.2*doubanrecloss1+0.2*weiborecloss1
            
    print(allloss)

    DEnoptimizer.zero_grad()
    WEnoptimizer.zero_grad()
    gnnnetoptimizer.zero_grad()
    allloss.backward()
    DEnoptimizer.step()
    WEnoptimizer.step()
    gnnnetoptimizer.step()

    if epoch>int(epochs*9/10):
        
        print('eval::::::::::::::::::::::at  ',epoch)

        doubanencoder.eval()
        weiboencoder.eval()
        gnnnet.eval()
        
        x=[piar[0] for piar in traindata]
        y=[piar[1] for piar in traindata]
        label=trainlabel
        
        h1 = doubanencoder(douban, doubanedges, doubanfeatureedges, douban_edge_dict)
        h2 = weiboencoder(weibo, weiboedges, weibofeatureedges, weibo_edge_dict)
        
        #doubanrecloss=recon_loss(product,h1,doubanedges,douban_neg_edgeindex_0).item()
        #weiborecloss=recon_loss(product,h2,weiboedges,weibo_neg_edgeindex_0).item()
        doubanrecloss=recon_loss(product,h1,doubanedges).item()
        weiborecloss=recon_loss(product,h2,weiboedges).item()
        doubanrecloss1=recon_loss1(product,h1,doubanedges1,doubanedges2).item()
        weiborecloss1=recon_loss1(product,h2,weiboedges1,weiboedges2).item()

        doubansample=h1[x]
        weibosample=h2[y]
        _,predictions=gnnnet(torch.cat((doubansample,weibosample),1))
        pred=predictions.cpu().detach().numpy().tolist()
        acc=accuracy_score( label, pred)
        f1=f1_score(label, pred, average='weighted')

        tmp_loss=0.2*doubanrecloss+0.2*weiborecloss+0.2*doubanrecloss1+0.2*weiborecloss1

        if acc>=(bestacc) and f1>=(bestf1) and tmp_loss<=bestloss:
            bestepoch=epoch
            bestacc=acc
            bestf1=f1
            bestloss=tmp_loss
            torch.save(doubanencoder.state_dict(),'./model/doubanencoder'+tag+'.pkl')
            torch.save(weiboencoder.state_dict(),'./model/weiboencoder'+tag+'.pkl')
            torch.save(gnnnet.state_dict(),'./model/gnnnet'+tag+'.pkl')
            print(epoch,bestacc,bestf1,bestloss)

print('VAL:::',bestepoch,bestacc,bestf1,bestloss)

doubanencoder_test= Encoder().cuda()
weiboencoder_test=Encoder().cuda()
gnnnet_test=AutoEncoder().cuda()

doubanencoder_test.load_state_dict(torch.load('./model/doubanencoder'+tag+'.pkl'))
weiboencoder_test.load_state_dict(torch.load('./model/weiboencoder'+tag+'.pkl'))
gnnnet_test.load_state_dict(torch.load('./model/gnnnet'+tag+'.pkl'))

doubanencoder_test.eval()
weiboencoder_test.eval()
gnnnet_test.eval()

x=[piar[0] for piar in testdata]
y=[piar[1] for piar in testdata]
label=testdatalabel

h1 = doubanencoder_test(douban, doubanedges, doubanfeatureedges, douban_edge_dict)
h2 = weiboencoder_test(weibo, weiboedges, weibofeatureedges, weibo_edge_dict)

#doubanrecloss=recon_loss(product,h1,doubanedges,douban_neg_edgeindex_0).item()
#weiborecloss=recon_loss(product,h2,weiboedges,weibo_neg_edgeindex_0).item()
doubanrecloss=recon_loss(product,h1,doubanedges).item()
weiborecloss=recon_loss(product,h2,weiboedges).item()
doubanrecloss1=recon_loss1(product,h1,doubanedges1,doubanedges2).item()
weiborecloss1=recon_loss1(product,h2,weiboedges1,weiboedges2).item()

doubansample=h1[x]
weibosample=h2[y]
_,predictions=gnnnet_test(torch.cat((doubansample,weibosample),1))

pred=predictions.cpu().detach().numpy().tolist()
res=str(bestepoch)+'  '
res+=str(doubanrecloss+weiborecloss+doubanrecloss1+weiborecloss1)+'  '
res+=str(accuracy_score( label, pred))+'  '
res+=str(f1_score(label, pred , average='weighted'))+'  '
print(res)

with open('./'+tag+'.txt','w') as f:
    f.write(str(res))
doubanembedding=h1.detach().cpu().numpy().tolist()
weiboembedding=h2.detach().cpu().numpy().tolist()
with open('./output/doubanembedding'+tag+'.txt','w') as f:
    f.write(str(doubanembedding))
with open('./output/weiboembedding'+tag+'.txt','w') as f:
    f.write(str(weiboembedding))