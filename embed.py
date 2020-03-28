import _pickle as pkl
from gensim.models import FastText
from gensim import utils
from gensim.models.fasttext import FastText as FT_gensim
from gensim.corpora import Dictionary
import pandas as pd
from preprocess import split_token
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET
import signal



def fastText(corpus):

    from configparser import ConfigParser
    cp = ConfigParser()
    cp.read('config.ini')
    size = int(cp.get('model', 'embed_size'))
    window = int(cp.get('fastText', 'window'))

    print("training fastText model")
    model = FastText(size=size, window=window, min_count=1)
    model.build_vocab(sentences=corpus)
    model.train(sentences=corpus, total_examples=len(corpus), epochs=10)

    return model

# 构造code以及对应的desc的corpus


def build_code_desc_corpus(path):  # 构建供fastText模型训练的corpus
    df = pd.read_json(path)
    codes = df['code']
    descs = df['description']

    assert len(codes) == len(descs)

    corpus_len = len(codes)
    corpus = []
    bar = tqdm(range(corpus_len))
    for i in bar:
        bar.set_description("build corpus")
        tmp = codes[i]+" "+descs[i]
        tokens = []
        for token in tmp.split():
            tokens += split_token(token)

        corpus.append(tokens)

    return corpus

# 构造ast中间节点的corpus
def build_ast_node_corpus(path, language):
    print("build ast nonterminal corpus")
    def generate_ast_seq(root, seq):  # 构造ast中间节点序列
        seq.append(root.tag)
        for child in root.getchildren():
            generate_ast_seq(child, seq)
        return seq 

    df = pd.read_json(path)
    xmls = df['xml']
    corpus = []
    parse_error_num = 0
    for xml in tqdm(xmls):
        try:
            root = ET.fromstring(xml)
            corpus.append(generate_ast_seq(root, []))
        except Exception:
            parse_error_num += 1
            continue

    print("parse error number: ", parse_error_num)

    return corpus


