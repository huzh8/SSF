import pickle
import numpy as np
import os
from tqdm import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile


def write_pickle(obj, outfile, protocol=-1):
    with open(outfile, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def read_pickle(infile):
    with open(infile, 'rb') as f:
        return pickle.load(f)


def doc2vec(docs, dir, epochs=50):
    docs_tagged = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]
    print('Initializing...')

    path = '{}/d2v_empty.pkl'.format(dir)
    if os.path.exists(path):
        model = read_pickle(path)
    else:
        print('building vocabulary...')
        model = Doc2Vec(vector_size=50, window=5,
                        min_count=2, workers=8)
        model.build_vocab(docs_tagged)
        write_pickle(model, path)
    path = '{}/d2v_pretrained_0.pkl'.format(dir)
    if os.path.exists(path):
        model = read_pickle(path)
    else:
        print('initializing empty model')
        model.train(docs_tagged,
                    total_examples=model.corpus_count,
                    epochs=0)
        write_pickle(model, path)
    path = '{}/d2v_pretrained_{}.pkl'.format(dir, epochs)
    if os.path.exists(path):
        model = read_pickle(path)
    else:
        print('training...')
        model.dbow_words = 0
        for i in range(50):
            print(i)
            model.train(docs_tagged,
                    total_examples=model.corpus_count,
                    epochs=1)
        write_pickle(model, path)
    print("finish traing")
    vec = model.docvecs
    dv = np.array([vec[i] for i in range(len(vec))]).tolist()
    with open('./doc2vec_tmp_50/docvecs.txt',"w") as f:
            f.write(str(dv))
    word_vectors = model.wv
    fname = get_tmpfile("vectors.kv")
    print(fname)
    word_vectors.save(fname)
    #return dv,mv


if __name__ == '__main__':
    # 把单个用户的多篇文档拼成一份文档
    # 建议把两个网络的文档放在一起训练，这样得到的文档向量落在同一个语义空间
    
    douban_words = []
    with open('./douban_words.txt', 'r',encoding='gbk') as f:
        douban_words = eval(f.read())

    weibo_words = []
    with open('./weibo_words.txt', 'r',encoding='gbk') as f:
        weibo_words = eval(f.read())
    
    doubanlenght=0
    for i in douban_words:
        for j in i:
            doubanlenght+=1
    weibolenght=0
    for i in weibo_words:
        for j in i:
            weibolenght+=1
    print(doubanlenght,weibolenght)