import numpy as np
import torch
import pandas as pd
from configparser import ConfigParser
from torch.utils.data import Dataset
import pandas as pd
import math
from tqdm import tqdm
from utils import cos_sim, ACC, MAP, NDCG, MRR, IDCG, buildLogger
from tensorboardX import SummaryWriter
import time
from datetime import datetime
import random
import logging
from gensim.models.fasttext import FastText as FT_gensim
from model import JointEncoderWithAttention

random.seed(12345)
np.random.seed(12345)
torch.manual_seed(12345)
torch.cuda.manual_seed(12345)


def pad(seqs, pad_index):  # pad data
    lens = [len(seq) for seq in seqs]
    max_len = max(lens)
    pad_seqs = [[pad_index]*(max_len-length) for length in lens]

    for i in range(len(seqs)):
        seqs[i].extend(pad_seqs[i])

    seqs = np.asarray(seqs)
    return seqs


def get_batch(dataset, idx, bs, pad_index):
    tmp = dataset.iloc[idx: idx+bs]
    code_tokens, ast_seq, desc_pos, desc_neg = [], [], [], []
    for _, item in tmp.iterrows():
        code_tokens.append(item[0])
        ast_seq.append(item[1])
        desc_pos.append(item[2])
        desc_neg.append(item[3])

    code_tokens_len = [len(tokens) for tokens in code_tokens]
    ast_seq_len = [len(seq) for seq in ast_seq]
    desc_pos_len = [len(desc) for desc in desc_pos]
    desc_neg_len = [len(desc) for desc in desc_neg]

    code_tokens = pad(code_tokens, pad_index)
    desc_pos = pad(desc_pos, pad_index)
    desc_neg = pad(desc_neg, pad_index)

    return [code_tokens, code_tokens_len, ast_seq, ast_seq_len, desc_pos, desc_pos_len, desc_neg, desc_neg_len]


def train(model, train_dataset, valid_dataset, pad_index, epochs, batch_size, pool_size, optimizer, clip, log_every, valid_every):
    print("start training")
    logger = buildLogger("log/train.log", "train")
    batch_step = 0  # record the number of processed batch
    timestamp = datetime.now().strftime('%Y-%m-%d-%H:%M')
    tb_writer = SummaryWriter(f'log/{timestamp}')

    max_valid_mrr = 0
    for epoch in range(epochs):
        itr_start_time = time.time()
        index = 0
        losses = []
        while(index < len(train_dataset)):
            model.train()
            batch = get_batch(train_dataset, index, batch_size, pad_index)
            index += batch_size

            optimizer.zero_grad()  # clear gradient
            loss = model(*batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # set threshold to avoid gradient vanish
            optimizer.step()  # update network paramenters

            losses.append(loss.item())

            if batch_step % log_every == 0 and batch_step :  # record loss
                elapsed = time.time()-itr_start_time
                logger.info("epoch:{} bath_step:{} step_time:{:.3f}s loss:{:.3f}".format(epoch, batch_step, elapsed, np.mean(losses)))

                if tb_writer:
                    tb_writer.add_scalar('loss', np.mean(losses), batch_step*batch_size)

                losses = []
                itr_start_time = time.time()

            if batch_step % valid_every == 0 and batch_step:  # validate
                print("validating....")
                acc, mrr, map, ndcg = validate(valid_dataset, model, pool_size, pad_index) 
                logger.info(f'acc:{acc},mrr:{mrr},map:{map},ndcg:{ndcg}')

                if tb_writer:
                    tb_writer.add_scalar('acc', acc, batch_step)
                    tb_writer.add_scalar('mrr', mrr, batch_step)
                    tb_writer.add_scalar('map', map, batch_step)
                    tb_writer.add_scalar('ndcg', ndcg, batch_step)

                if mrr > max_valid_mrr:  # update best model
                    logger.info(f"max_valid_mrr{mrr}")
                    max_valid_mrr = mrr
                    torch.save(model.state_dict(), "log/code_search.pt")

            batch_step += 1


def validate(valid_dataset, model, pool_size, pad_index):
    """
    simple validation in a code pool.
    @param: poolsize - size of the code pool, if -1, load the whole test set
    """

    model.eval()
    processd_num = 0  # record the number of processed data
    accs, mrrs, maps, ndcgs = [], [], [], []
    code_reprs, desc_reprs = [], []
    while processd_num < len(valid_dataset)-batch_size:
        # batch:code_tokens, code_tokens_len, ast_seq, ast_seq_len, desc_pos, desc_pos_len, desc_neg, desc_neg_len
        batch = get_batch(valid_dataset, processd_num, batch_size, pad_index)
        processd_num += batch_size

        code_batch = batch[:4]
        desc_batch = batch[4:6]

        with torch.no_grad():
            code_repr = model.code_encode(*code_batch).data.cpu().numpy().astype(np.float32)
            desc_repr = model.desc_encode(*desc_batch).data.cpu().numpy().astype(np.float32)

        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)

    code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)

    assert len(code_reprs) == len(desc_reprs)

    bar = tqdm(range(0, len(code_reprs), pool_size))
    bar.set_description("start valid")
    for k in bar:
        if k+pool_size >len(bar):
            break
        code_matrix = code_reprs[k:k+pool_size]
        desc_matrix = desc_reprs[k:k+pool_size]
        real = list(range(pool_size))
        sims = cos_sim(desc_matrix, code_matrix)  # use description to search code
        negsim = np.negative(sims)
        predict = np.argpartition(negsim, kth=pool_size-1)
        predict=predict[:pool_size]

        for i in range(len(real)):
            accs.append(ACC([real[i]], list(predict[i])))
            mrrs.append(MRR([real[i]], list(predict[i])))
            maps.append(MAP([real[i]], list(predict[i])))
            ndcgs.append(NDCG([real[i]], list(predict[i])))

    return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)


if __name__ == "__main__":
    model = FT_gensim.load('dataset/C#/fastText/fastText')
    pad_index = model.wv.vocab['<pad>'].index
    config = ConfigParser()
    config.read('config.ini')

    # model parameters
    embed_size = config.getint('model', 'embed_size')
    use_gpu = config.getint('model', 'use_gpu')
    batch_size = config.getint('model', 'batch_size')
    rnn_hidden_size = config.getint('model', 'rnn_hidden_size')
    tree_embed_size = config.getint('model', 'tree_embed_size')
    code_ast_repr_size = config.getint('model', 'code_ast_repr_size')
    code_token_repr_size = config.getint('model', 'code_token_repr_size')
    code_combine_repr_size = config.getint('model', 'code_combine_repr_size')
    desc_repr_size = config.getint('model', 'desc_repr_size')
    margin=config.getfloat('model','margin')
    sim_measure=config.get('model','sim_measure')
    vocab_size = model.wv.syn0.shape[0]

    pretrained_weight = np.zeros((model.wv.syn0.shape[0], model.wv.syn0.shape[1]), dtype="float32")
    pretrained_weight[:model.wv.syn0.shape[0]] = model.wv.syn0

    # train parameters
    learning_rate = config.getfloat('train', 'learning_rate')
    adam_epsilon = config.getfloat('train', 'adam_epsilon')
    clip = config.getfloat('train', 'clip')
    
    log_every = config.getint('train', 'log_every')
    valid_every = config.getint('train', 'valid_every')
    epochs = config.getint('train', 'epoch')
    pool_size = config.getint('train', 'pool_size')

    train_dataset = pd.read_pickle("dataset/C#/input/train.pkl")
    valid_dataset = pd.read_pickle('dataset/C#/input/valid.pkl')

    model = JointEncoderWithAttention(batch_size, embed_size, use_gpu, rnn_hidden_size,
                                      tree_embed_size, code_ast_repr_size, code_token_repr_size, code_combine_repr_size, desc_repr_size, vocab_size, pretrained_weight,margin,sim_measure)
    if use_gpu:
        model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
    train(model, train_dataset, valid_dataset, pad_index, epochs, batch_size, pool_size, optimizer, clip, log_every, valid_every)
