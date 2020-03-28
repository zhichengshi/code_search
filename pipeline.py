import os
import pandas as pd
from xml.etree import ElementTree as ET
from preprocess import *
from embed import *
from gensim.models.fasttext import FastText as FT_gensim
import random
import numpy as np 

class Pipeline():
    def __init__(self, root, language, part, vocab=None):
        self.root = root
        self.language = language
        self.part = part

        self.vocab = vocab

        self.desc_code_path = None
        self.desc_code_pkl_path = None

    def embed(self):
        print("start embed")
        path = os.path.join(self.root, self.language, 'xml', self.part+".json")  # train code and description embeddings
        corpus = build_code_desc_corpus(path)
        model = fastText(corpus)

        corpus = build_ast_node_corpus(path, self.language)  # train ast nonterminal token embedding
        model.build_vocab(sentences=corpus, update=True)
        model.train(sentences=corpus, total_examples=len(corpus), epochs=10)

        dump_path = os.path.join(self.root, self.language, "fastText", "fastText")  # dump

        entities = ["<pad>"] # append pad 
        embedding_size=model.wv.syn0.shape[1]
        weights = np.random.normal(size=(len(entities), embedding_size))
        model.wv.add(entities=entities, weights=weights, replace=False)

        model.save(dump_path)
        self.vocab=model.wv.vocab

    def splitAst(self):  # code tokens, code asts, description
        print("start generate input data")
        path = os.path.join(self.root, self.language, 'xml', self.part+".json")

        if not self.vocab:
            model_path = os.path.join(self.root, self.language, 'fastText', 'fastText')
            self.vocab = FT_gensim.load(model_path).wv.vocab

        df = pd.read_json(path)
        code_tokens = []
        ast_seqs = []
        desc_pos = []
        desc_neg = []

        descs = df['description']

        for _, row in df.iterrows():
            code = row['code']
            xml = row['xml']
            desc = row['description']

            try:
                ast_seqs.append(xml2astSeq(xml, self.vocab, self.language))
                code_tokens.append(seq2id(code, self.vocab))
                desc_pos.append(seq2id(desc, self.vocab))
                rand_index = random.randint(0, len(descs)-1)  # generate negative sample
                desc_neg.append(seq2id(descs[rand_index], self.vocab))
            except:
                continue

        # 写出
        dump_path = os.path.join(self.root, self.language, 'input', self.part+'.pkl')
        df=pd.DataFrame()
        df['code_tokens'] = code_tokens
        df['ast_seq'] = ast_seqs
        df['desc_pos'] = desc_pos
        df['desc_neg'] = desc_neg
        df.to_pickle(dump_path)


if __name__ == "__main__":
    root = 'dataset'
    language = 'C#'
    part = 'train'
    ppl = Pipeline(root, language, part)
    # ppl.embed() # only use the training set to train fastText model
    ppl.splitAst()
    removeZeroLenData(root,language,part)

    part='valid'
    ppl = Pipeline(root, language, part)
    ppl.splitAst()
    removeZeroLenData(root,language,part)
    
    part='test'
    ppl = Pipeline(root, language, part)
    ppl.splitAst()
    removeZeroLenData(root,language,part)

