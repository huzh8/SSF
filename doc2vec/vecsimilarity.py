import numpy as np
import torch
import pickle
from tqdm import tqdm

def normalize_features(x):
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
    x = x / x_norm
    return x

def normalize_features1(input_data):
    MIN,MAX=np.min(input_data),np.max(input_data)
    input_data= (input_data-MIN)/(MAX-MIN)
    return input_data

def similarity(a_vect, b_vect):
    """计算两个向量余弦值
    """
    dot_val = 0.0
    a_norm = 0.0
    b_norm = 0.0
    cos = None
    for a, b in zip(a_vect, b_vect):
        dot_val += a*b
        a_norm += a**2
        b_norm += b**2
    if a_norm == 0.0 or b_norm == 0.0:
        cos = -1
    else:
        cos = dot_val / ((a_norm*b_norm)**0.5)
    return cos
    
'''
#先通过注释部分代码生成嵌入两两间的相似度。
print('start reading')
sentences = []
with open('./doc2vec_tmp/docvecs.txt','r') as fileTrainRaw:
    d=fileTrainRaw.read()
    sentences=eval(d)
douban=sentences[0:4399]
weibo=sentences[4399:]
douban=normalize_features(douban)
weibo=normalize_features(weibo)
douban=torch.FloatTensor(douban)
weibo=torch.FloatTensor(weibo)

tmp=[]
for i in tqdm(douban):
    start=i.repeat(4399, 1)
    distribution = torch.cosine_similarity(start, douban,dim=1).cpu().detach().numpy()
    distribution=normalize_features1(distribution).tolist()
    tmp.append(distribution)
with open('./doubansim.txt', 'wb') as f:
    pickle.dump(tmp, f)

tmp=[]
for i in tqdm(weibo):
    start=i.repeat(4414, 1)
    distribution = torch.cosine_similarity(start, weibo,dim=1).cpu().detach().numpy()
    distribution=normalize_features1(distribution).tolist()
    tmp.append(distribution)
with open('./weibosim.txt', 'wb') as f:
    pickle.dump(tmp, f)

'''
with open('./doubansim.txt', 'rb') as f:
    tmp = pickle.load(f)

print('end read')
douban_similarity=np.zeros((4399,4399))
k1=0.457
#k1=0.45
for index1,i in enumerate(tmp):
    #print(index1)
    for index2,j in enumerate(i):
        if j>=k1:
            douban_similarity[index1][index2]=1
print('1')

with open('./weibosim.txt', 'rb') as f:
    tmp = pickle.load(f)
print('end read')
weibo_similarity=np.zeros((4414,4414))
k2=0.56
#k2=0.548
for index1,i in enumerate(tmp):
    #print(index1)
    for index2,j in enumerate(i):
        if j>=k2:
            weibo_similarity[index1][index2]=1
print('2')
doubanpiar=[]
for i in range(4399):
    for j in range(4399):
        if douban_similarity[i][j]==1 and i!=j:
            doubanpiar.append([i,j])

weibopiar=[]
for i in range(4414):
    for j in range(4414):
        if weibo_similarity[i][j]==1 and i!=j:
            weibopiar.append([i,j])

print(len(doubanpiar),len(weibopiar))
with open('./doubansimpiar.txt','w') as f:
    f.write(str(doubanpiar))
with open('./weibosimpiar.txt','w') as f:
    f.write(str(weibopiar))
