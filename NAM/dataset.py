import numpy as np
import random
np.random.seed(0)
random.seed(0)
class Batch():
    def __init__(self,data,batch_size,all_size):
        self.data=data
        self.batch_size=batch_size
        self.batch_index=0
        self.all_size=all_size
    def getbatch(self):
        if self.batch_index<self.all_size-self.batch_size:
            batch=self.data[self.batch_index:self.batch_index+self.batch_size]
            self.batch_index=self.batch_index+self.batch_size
            return batch
        else:
            tmp=self.batch_index+self.batch_size-self.all_size
            batch=self.data[self.batch_index:self.all_size]+self.data[0:tmp]
            self.batch_index=tmp
            return batch

class Getdata():
    def __init__(self):
        with open('../dataset/train-0.05.txt',"r") as f:
            self.train=eval(f.read())
        with open('../dataset/test-0.05.txt',"r") as f:
            self.test=eval(f.read())
        with open('../dataset/val-0.05.txt',"r") as f:
            self.val=eval(f.read())
        
        print(len(self.train),len(self.test),len(self.val))
        self.train_i=[]
        self.train_j=[]
        for i in range(4399):
            self.train_i.append(i)
        for j in range(4414):
            self.train_j.append(j)


    def gettrain(self):
        test=[]
        for index,i in enumerate(self.train):
            test.append(i)
        testlabel=np.ones(len(test),dtype=np.int).tolist()

        negtest=[]
        for i,j in test:
            rand_index = np.random.randint(2)
            j_index=np.random.choice(self.train_j)
            #print(j_index)
            while j_index ==j :
                print('好巧！')
                j_index=np.random.choice(self.train_j)
            i_index=np.random.choice(self.train_i)
            while i_index ==i :
                print('好巧！')
                i_index=np.random.choice(self.train_i)
            if rand_index==0:
                negtest.append([i,j_index])
            elif rand_index==1:
                negtest.append([i_index,j])
        
        negtest=list(set([tuple(t) for t in negtest]))
        negtest=np.array(negtest).tolist()
        neg_testlabel=np.zeros(len(negtest),dtype=np.int).tolist()
        
        finaltest=test+negtest
        finaltestlabel=testlabel+neg_testlabel
        #finaltestlabel=[int(i) for i in finaltestlabel]
        state = np.random.get_state()
        np.random.shuffle(finaltest)
        np.random.set_state(state)
        np.random.shuffle(finaltestlabel)
        #print(len(finaltest),len(finaltestlabel))
        return finaltest,finaltestlabel
        

    def gettest(self):
        test=[]
        for index,i in enumerate(self.test):
            test.append(i)
        testlabel=np.ones(len(test),dtype=np.int).tolist()

        negtest=[]
        for i,j in test:
            rand_index = np.random.randint(2)
            j_index=np.random.choice(self.train_j)
            while j_index ==j :
                print('好巧！')
                j_index=np.random.choice(self.train_j)
            #negtest.append([i,j_index])
            i_index=np.random.choice(self.train_i)
            while i_index ==i :
                print('好巧！')
                i_index=np.random.choice(self.train_i)
            #negtest.append([i_index,j])
            if rand_index==0:
                negtest.append([i,j_index])
            elif rand_index==1:
                negtest.append([i_index,j])
        
        negtest=list(set([tuple(t) for t in negtest]))
        negtest=np.array(negtest).tolist()
        neg_testlabel=np.zeros(len(negtest),dtype=np.int).tolist()
        
        finaltest=test+negtest
        finaltestlabel=testlabel+neg_testlabel
        state = np.random.get_state()
        np.random.shuffle(finaltest)
        np.random.set_state(state)
        np.random.shuffle(finaltestlabel)
        return finaltest,finaltestlabel

    def getval(self):
        test=[]
        for index,i in enumerate(self.val):
            test.append(i)
        testlabel=np.ones(len(test),dtype=np.int).tolist()

        negtest=[]
        for i,j in test:
            rand_index = np.random.randint(2)
            j_index=np.random.choice(self.train_j)
            while j_index ==j :
                print('好巧！')
                j_index=np.random.choice(self.train_j)
            #negtest.append([i,j_index])
            i_index=np.random.choice(self.train_i)
            while i_index ==i :
                print('好巧！')
                i_index=np.random.choice(self.train_i)
            #negtest.append([i_index,j])
            if rand_index==0:
                negtest.append([i,j_index])
            elif rand_index==1:
                negtest.append([i_index,j])

        negtest=list(set([tuple(t) for t in negtest]))
        negtest=np.array(negtest).tolist()
        neg_testlabel=np.zeros(len(negtest),dtype=np.int).tolist()
        
        finaltest=test+negtest
        finaltestlabel=testlabel+neg_testlabel
        #finaltestlabel=[int(i) for i in finaltestlabel]
        state = np.random.get_state()
        np.random.shuffle(finaltest)
        np.random.set_state(state)
        np.random.shuffle(finaltestlabel)
        #print(len(finaltest),len(finaltestlabel))
        return finaltest,finaltestlabel

    def getrealtrain(self):
        test=[]
        for index,i in enumerate(self.train):
            test.append(i)
        return test

def normalize_features(x):
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
    x = x / x_norm
    return x
