import numpy as np
import time
import math
import torch
from torch.nn import functional as F
import logging
import sys 
import pandas as pd 
import os

def normalize(data):
    """normalize matrix by rows"""
    return data/np.linalg.norm(data,axis=1,keepdims=True)

def dot_np(data1,data2):
    """cosine similarity for normalized vectors"""
    #print("warning: the second matrix will be transposed, so try to put the simpler matrix as the second argument in order to save time.")
    return np.dot(data1, data2.T)

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def similarity(code_vec, desc_vec, measure='cos'):
    if measure=='cos':
        vec1_norm = normalize(code_vec)
        vec2_norm = normalize(desc_vec)
        return np.dot(vec1_norm, vec2_norm.T)[:,0]
    elif measure=='poly':
        return (0.5*np.dot(code_vec, desc_vec.T).diagonal()+1)**2
    elif measure=='sigmoid':
        return np.tanh(np.dot(code_vec, desc_vec.T).diagonal()+1)
    elif measure in ['enc', 'gesd', 'aesd']: #https://arxiv.org/pdf/1508.01585.pdf 
        euc_dist = np.linalg.norm(code_vec-desc_vec, axis=1)
        euc_sim = 1 / (1 + euc_dist)
        if measure=='euc': return euc_sim                
        sigmoid_sim = sigmoid(np.dot(code_vec, desc_vec.T).diagonal()+1)
        if measure == 'gesd': return euc_sim * sigmoid_sim
        elif measure == 'aesd': return 0.5*(euc_sim+sigmoid_sim)

def cos_sim(data1,data2):
    """numpy implementation of cosine similarity for matrix"""
    #print("warning: the second matrix will be transposed, so try to put the simpler matrix as the second argument in order to save time.")
    dotted = np.dot(data1,np.transpose(data2))
    norm1 = np.linalg.norm(data1,axis=1)
    norm2 = np.linalg.norm(data2,axis=1)
    matrix_vector_norms = np.multiply(norm1, norm2)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors


def ACC(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum+1
    return sum/float(len(real))

def MAP(real, predict):
    sum = 0.0
    for id, val in enumerate(real):
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum+(id+1)/float(index+1)
    return sum/float(len(real))

def MRR(real, predict):
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum+1.0/float(index+1)
    return sum/float(len(real))

def NDCG(real, predict):
    dcg = 0.0
    idcg = IDCG(len(real))
    for i, predictItem in enumerate(predict):
        if predictItem in real:
            itemRelevance = 1
            rank = i+1
            dcg += (math.pow(2, itemRelevance)-1.0)*(math.log(2)/math.log(rank+1))
    return dcg/float(idcg)

def IDCG(n):
    idcg = 0
    itemRelevance = 1
    for i in range(n):
        idcg += (math.pow(2, itemRelevance)-1.0)*(math.log(2)/math.log(i+2))
    return idcg


def buildLogger(log_file,part):
    logger = logging.getLogger(part)
    logger.setLevel(level=logging.DEBUG)
    
    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.INFO)
    logger.addHandler(stream_handler)
    
    # FileHandler
    file_handler = logging.FileHandler(log_file,mode='w')
    file_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def removeZeroLenData(root,language,part): # remove the data which length is 0
    path=os.path.join(root,language,part+".pkl")
    df=pd.read_pickle(path)
    code_tokens, ast_seq, desc_pos, desc_neg = [], [], [], []
    lengths=set()
    cnt=0
    for _,item in df.iterrows():
        for data in item:
            lengths.add(len(data))

        if 0 in lengths:
            cnt+=1
            lengths.clear()
            continue
        else:
            code_tokens.append(item[0])
            ast_seq.append(item[1])
            desc_pos.append(item[2])
            desc_neg.append(item[3])
    print("filter rows: ",cnt)
    dump_df=pd.DataFrame()
    dump_df['code_tokens']=code_tokens
    dump_df['ast_seq']=ast_seq
    dump_df['desc_pos']=desc_pos
    dump_df['desc_neg']=desc_neg

    dump_path='dataset/C#/input/'+part+'.pkl'
    dump_df.to_pickle(dump_path)