import os 
os.environ['CUDA_VISIBLE_DEVICES']='7'
import argparse
import os
import numpy as np
import math
import sys
import pickle
from wgan_evaluate import *
from torch.autograd import Variable
from tqdm import *
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import time
import random
from spectral_normalization import SNLinear
from dataset import *

def normalize_features(input_data):
    input_data=np.array(input_data)
    length=len(input_data)
    for i in range(0,length):
        MIN,MAX=np.min(input_data[i]),np.max(input_data[i])
        if MIN!=MAX:
            input_data[i]= (input_data[i]-MIN)/(MAX-MIN)
    output_data=input_data.tolist()
    return output_data

def normalize_features1(x):
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
    x = x / x_norm
    return x

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()            
parser.add_argument("--n_epochs", type=int, default=20000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
opt = parser.parse_args()

with open('../FEM/output/doubanembedding-pygdhgnn-50000-0.05.txt',"r") as f:
    douban=eval(f.read())
with open('../FEM/output/weiboembedding-pygdhgnn-50000-0.05.txt',"r") as f:
    weibo=eval(f.read())

X=douban
Y=weibo

X=normalize_features1(X)
Y=normalize_features1(Y)
X_id=[]
for i in range(len(X)):
    X_id.append(i)
Y_id=[]
for i in range(len(Y)):
    Y_id.append(i)

X_degree_1=np.ones((len(X_id)))
X_degree_1=X_degree_1/np.sum(X_degree_1)
Y_degree_1=np.ones((len(Y_id)))
Y_degree_1=Y_degree_1/np.sum(Y_degree_1)

data=Getdata()
train=data.getrealtrain()

xxx=[]
yyy=[]
for i,j in train:
    xxx.append(X[i])
    yyy.append(Y[j])
yyy_all=Tensor(Y)
##################

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model=nn.Linear(100,100,bias=False)
        self.model.weight.data=torch.eye(100)
    def forward(self, z):
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            SNLinear(100, 50),
            #nn.BatchNorm1d(50),
            nn.LeakyReLU(0.1),
            SNLinear(50, 1),
            #nn.BatchNorm1d(1),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        validity = self.model(x)
        return validity

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

timestamp=time.time()
result=[]
for epoch in trange(int(opt.n_epochs)):
        generator.train()
        discriminator.train()
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()
        S=[]
        T=[]
        S_sample=np.random.choice(X_id, opt.batch_size, replace=True, p=X_degree_1)
        T_sample=np.random.choice(Y_id, opt.batch_size, replace=True, p=Y_degree_1)
        for i in S_sample:
            S.append(X[i])
        for i in T_sample:
            T.append(Y[i])
        S_sample=Tensor(S)
        T_sample=Tensor(T)

        fake_sample = generator(S_sample)
        real_validity = discriminator(T_sample)
        fake_validity = discriminator(fake_sample)

        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        d_loss.backward()
        optimizer_D.step()
        
        if epoch % 10 ==3 or  epoch % 10 ==8 or  epoch % 10 ==1 or  epoch % 10 ==6:
            generator.eval()
            discriminator.eval()

            fake=generator(Tensor(X))
            real_validity = discriminator(fake)
            fake_validity = discriminator(Tensor(Y))
            
            max1=torch.max(real_validity)
            max2=torch.max(fake_validity)
            
            real_validity=real_validity-max1
            real_validity=real_validity*5
            fake_validity=fake_validity-max2
            fake_validity=fake_validity*5
           
            X_degree_1=F.softmax(real_validity,dim=0)
            Y_degree_1=F.softmax(fake_validity,dim=0)
            
            X_degree_1=torch.squeeze(X_degree_1)
            Y_degree_1=torch.squeeze(Y_degree_1)
            
            X_degree_1=X_degree_1.cpu().detach().numpy()
            Y_degree_1=Y_degree_1.cpu().detach().numpy()
        
        if epoch % opt.n_critic ==0 :
            generator.train()
            discriminator.train()
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            S=[]
            T=[]
            S_sample=np.random.choice(X_id, opt.batch_size, replace=True, p=X_degree_1)
            T_sample=np.random.choice(Y_id, opt.batch_size, replace=True, p=Y_degree_1)
            for i in S_sample:
                S.append(X[i])
            for i in T_sample:
                T.append(Y[i])
            S_sample=Tensor(S)
            T_sample=Tensor(T)

            fake_imgs = generator(S_sample)
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            # orthogonal loss
            o_loss=torch.mean(torch.abs(S_sample-torch.mm(fake_imgs,generator.model.weight.data.t())))
            #弱监督信息
            xxx_sample=generator(Tensor(xxx))
            yyy_sample=Tensor(yyy)
            ooo_loss=torch.mean(torch.cosine_similarity(xxx_sample,yyy_sample))
            
            g_loss =-torch.mean(fake_validity)+0.5*o_loss-ooo_loss
            g_loss.backward()
            optimizer_G.step()
        if epoch % 100 ==0 and epoch!= 0:    
            print(
                "[Epoch %d/%d]  [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, d_loss.item(), g_loss.item())
            )

        if epoch % 500 ==0: #and epoch!= 0: 
            generator.eval()
            fake=generator(Tensor(X))
            #print(timestamp)
            evalu=evaluate(X_id,Y_id,len(X_id),len(Y_id))
            evalu.getcosfiles(fake,Tensor(Y),timestamp)
            res=evalu.report(timestamp)
            result.append(str(epoch)+str(res)+'   ')
            print(res)

with open('./result/result-dhgnn-50000-0.05-10.txt','w') as f:
    f.write(str(result))
