#from gensim import corpora
from collections import defaultdict
from tqdm import *
import pickle
import numpy as np

douban_words = []
with open('./splitword/douban_words.txt', 'rb') as f:
    douban_words = pickle.load(f)

weibo_words = []
with open('./splitword/weibo_words.txt', 'rb') as f:
    weibo_words = pickle.load(f)
#print(douban_words[3636],weibo_words[3636])
print('end read')
frequency = defaultdict(int)
for texts in tqdm(douban_words+weibo_words):
    for text in texts:
        for token in text:
            frequency[token] += 1
douban=douban_words
weibo=weibo_words

#去掉词频为1的词
new_douban=[]
for texts in tqdm(douban_words):
    tmp = [[token for token in text if frequency[token] > 1]
        for text in texts]
    new_douban.append(tmp)

#去掉词频为1的词
new_weibo=[]
for texts in tqdm(weibo_words):
    tmp = [[token for token in text if frequency[token] > 1]
        for text in texts]
    new_weibo.append(tmp)
'''
'''
#去掉只有一个词的句子
douban=[]
for node in tqdm(new_douban):
    new_node=[]
    for sentence in node:
        if len(sentence)<2:
            continue
        else:
            new_node.append(sentence)
    douban.append(new_node)
#print(douban[0])
weibo=[]
for node in tqdm(new_weibo):
    new_node=[]
    for sentence in node:
        if len(sentence)<2:
            continue
        else:
            new_node.append(sentence)
    weibo.append(new_node)
#print(weibo[0])

for index,i in enumerate(douban):
    signl=0
    for j in i:
        if len(j)!=0:
            signl+=1
    if signl<=2:
        print('douban',index,douban[index])
for index,i in enumerate(weibo):
    signl=0
    for j in i:
        if len(j)!=0:
            signl+=1
    if signl<=2:
        print('weibo',index,weibo[index])

with open('./douban_words.pkl', 'wb') as f:
    pickle.dump(douban, f)
with open('./weibo_words.pkl', 'wb') as f:
    pickle.dump(weibo, f)

with open('./douban_words.txt','w') as f:
    f.write(str(douban))
with open('./weibo_words.txt','w') as f:
    f.write(str(weibo))
print('finish')