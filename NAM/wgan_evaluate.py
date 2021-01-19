import numpy as np
from tqdm import *
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
class evaluate():
    def __init__(self,x,y,length1,length2):
        self.x=x
        self.y=y
        self.length1=length1
        self.length2=length2

    def cos(self,target, behavior):
        target=target.repeat(len(behavior), 1)
        distribution = torch.cosine_similarity(target, behavior,dim=1).cpu().detach().numpy()
        distribution=self.normalize_features(distribution)
        attention_distribution = torch.FloatTensor(distribution)
        return attention_distribution

    def normalize_features(self,input_data):
        MIN,MAX=np.min(input_data),np.max(input_data)
        input_data= (input_data-MIN)/(MAX-MIN)
        return input_data

    def getcosfiles(self,X,Y,timestamp):
        Y=Y.detach()
        X=X.detach()
        count=0
        for i in tqdm(X):
            a=self.cos(i,Y).numpy().tolist()
            #因为涉及文件夹名称，所以要重用的话要改这里
            with open('./wgancache/'+str(timestamp)+str(self.x[count])+'.txt',"w") as f:
                f.write(str(a))
            count=count+1

    def report(self,timestamp):
        k=10
        topk=[]
        for i in self.x:
            b=[]
            with open('./wgancache/'+str(timestamp)+str(i)+'.txt',"r") as f:
                d = f.read()
                b=eval(d)
            tensor=torch.FloatTensor(b)
            emb, pred = torch.topk(tensor, k, dim=0, largest=True, sorted=True, out=None)
            topk.append(pred)
        
        truth=[]
        with open('../dataset/gt.txt',"r") as f:
            truth=eval(f.read())
        hit=0
        hit2=0
        for i in self.x:
            for index,j in enumerate(topk[i]):
                if [i,j] in truth:
                    hit=hit+((k-((index+1)-1))/k)
                    hit2=hit2+1
        return str(str(hit/len(truth))+'   '+str(hit2))
        