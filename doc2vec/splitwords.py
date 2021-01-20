#import jieba
import os
import re
from tqdm import *
import numpy as np 
import pickle
import sys
import importlib

#去掉所有不为中文的词
def process(fileTrainSeg):
    count = 1
    cn_reg = '^[\u4e00-\u9fa5]+$'
    for sentences in trange(len(fileTrainSeg)):
        for sentence in range(len(fileTrainSeg[sentences])):
            for words in range(len(fileTrainSeg[sentences][sentence])):
                if len(fileTrainSeg[sentences][sentence][words])!=0:
                    line_list = str(fileTrainSeg[sentences][sentence][words])
                    line_list_new = ''
                    for word in line_list:
                        if re.search(cn_reg, word):
                            line_list_new=line_list_new+str(word)
                    fileTrainSeg[sentences][sentence][words]=line_list_new
                    count += 1
                if count % 10000 == 0:
                    print('目前已处理%d个词' % count)
    newlist=[]
    for sentences in trange(len(fileTrainSeg)):
        newsentences=[]
        for sentence in range(len(fileTrainSeg[sentences])):
            if len(fileTrainSeg[sentences][sentence])!=0:
                newsentence=[]
                for words in range(len(fileTrainSeg[sentences][sentence])):
                    if len(fileTrainSeg[sentences][sentence][words])!=0:
                        newsentence.append(fileTrainSeg[sentences][sentence][words])
                if len(newsentence)==0:
                    print('error')
                else:
                    newsentences.append(newsentence)
        newlist.append(newsentences)

    return newlist


fileSegWordDonePath1 ='./splitword/douban_words.txt'
fileSegWordDonePath2 ='./splitword/weibo_words.txt'

# 读取文件内容到列表
douban = []
with open('../dataset_tmp/doubancontent.pkl', 'rb') as f:
    douban = pickle.load(f)

x=douban
final=process(x)

weibo = []
with open('../dataset_tmp/weibocontent.pkl', 'rb') as f:
    weibo = pickle.load(f)

y=weibo
final2=process(y)


with open(fileSegWordDonePath1, 'wb') as f:
    pickle.dump(final, f)

with open(fileSegWordDonePath2, 'wb') as f:
    pickle.dump(final2, f)




