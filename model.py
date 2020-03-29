import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


# 递归神经网络对树编码
class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.W_l = nn.Linear(encode_dim, encode_dim)
        self.W_r = nn.Linear(encode_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            self.embedding.weight.requires_grad = True

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            if node[i][0] is not -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                c_num = len(temp)
                for j in range(c_num):
                    if temp[j][0] is not -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

        batch_current = self.W_c(batch_current.index_copy(0, Variable(self.th.LongTensor(index)),
                                                          self.embedding(Variable(self.th.LongTensor(current_node)))))

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(0, Variable(self.th.LongTensor(children_index[c])), tree)
        # batch_current = F.tanh(batch_current)
        batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]

# 完整的astnn网络


class TreeEncoder(nn.Module):
    def __init__(self, embed_size, rnn_hidden_size, vocab_size, tree_embed_size, batch_size, use_gpu=True, pretrained_weight=None):
        super(TreeEncoder, self).__init__()
        self.stop = [vocab_size-1]
        self.hidden_dim = rnn_hidden_size
        self.hidden_dim = rnn_hidden_size
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embed_size
        self.encode_dim = tree_embed_size
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim,
                                        self.batch_size, self.gpu, pretrained_weight)
        # rnn
        self.biRnn = nn.LSTM(self.encode_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=True,
                             batch_first=True)
        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.1)

    def init_hidden(self):
        if self.gpu:
            if isinstance(self.biRnn, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size,  self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return (h0, c0)
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def forward(self, x):
        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            if max_len-lens[i]:
                seq.append(self.get_zeros(max_len-lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)

        # rnn
        output, hidden = self.biRnn(encodes, self.hidden)

        return output

# 对句子进行编码


class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, rnn_hidden_size, voc_pretrained_weight=None, use_gpu=1, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = rnn_hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, rnn_hidden_size, batch_first=True, bidirectional=True)
        self.voc_pretrained_weight = voc_pretrained_weight
        self.th = torch.cuda if use_gpu else torch

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.copy_(torch.from_numpy(self.voc_pretrained_weight))
        self.embedding.weight.requires_grad = True
        for name, param in self.lstm.named_parameters():  # initialize the gate weights
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)

    def forward(self, inputs, input_lens=None):
        (batch_size, seq_len) = inputs.shape
        inputs = self.embedding(Variable(self.th.LongTensor(inputs)).cuda())  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        inputs = F.dropout(inputs, 0.1, self.training)
        input_lens = Variable(self.th.LongTensor(input_lens)).cuda()

        if input_lens is not None:  # sort and pack sequence
            input_lens_sorted, indices = input_lens.sort(descending=True)
            inputs_sorted = inputs.index_select(0, indices)
            inputs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)

        output, (h_n, c_n) = self.lstm(inputs)  # hids:[b x seq x hid_sz*2](biRNN)

        if input_lens is not None:  # reorder and pad
            _, inv_indices = indices.sort()
            output, lens = pad_packed_sequence(output, batch_first=True)
            output = F.dropout(output, p=0.1, training=self.training)
            output = output.index_select(0, inv_indices)

        return output   # bach_size, seq_len,  n_dirs * hidden_size

# 带有atten的整个network


class JointEncoderWithAttention(nn.Module):
    def __init__(self, batch_size, embed_size, use_gpu, rnn_hidden_size, tree_embed_size,
                 code_ast_repr_size, code_token_repr_size, code_combine_repr_size, desc_repr_size, vocab_size, pretrained_weight, margin,sim_measure='cos'):
        super(JointEncoderWithAttention, self).__init__()

        self.sim_measure = sim_measure
        self.margin=margin
        # 构造三个编码器
        self.tree_encoder = TreeEncoder(embed_size, rnn_hidden_size, vocab_size, tree_embed_size, batch_size, use_gpu, pretrained_weight)
        self.token_encoder = SeqEncoder(vocab_size, embed_size, rnn_hidden_size, pretrained_weight)
        self.desc_encoder = SeqEncoder(vocab_size, embed_size, rnn_hidden_size, pretrained_weight)
        

        # 不同网络的线性映射
        self.tree_linear = nn.Linear(2*rnn_hidden_size, code_ast_repr_size)
        self.token_linear = nn.Linear(2*rnn_hidden_size, code_token_repr_size)
        self.combine_linear = nn.Linear(code_ast_repr_size+code_token_repr_size, code_combine_repr_size)
        self.desc_linear = nn.Linear(2*rnn_hidden_size, desc_repr_size)

        # 标准正态分布初始化attention变量
        self.tree_attn_param = nn.Parameter(torch.randn(code_ast_repr_size))
        self.token_attn_param = nn.Parameter(torch.randn(code_token_repr_size))
        self.desc_attn_param = nn.Parameter(torch.randn(desc_repr_size))

    # 初始化模型参数
    def init_weights(self):
        for linear in [self.tree_linear, self.token_linear,self.combine_linear, self.desc_linear]:
            linear.weight.data.uniform_(-0.1, 0.1)  # nn.init.xavier_normal_(m.weight)
            nn.init.constant_(linear.bias, 0.)

    # 对code部分进行编码
    def code_encode(self, tokens, token_len, asts, ast_len):
        batch_size = tokens.shape[0]
        token_mask=self.create_mask(token_len)
        ast_mask=self.create_mask(ast_len)
        # 处理code_token
        token_rnn_output = self.token_linear(self.token_encoder(tokens, token_len)).permute(0, 2, 1)  # (batch_size,repr_size,max_len)
        token_attn_param = self.token_attn_param.repeat(batch_size, 1).unsqueeze(1)  # (batch_size,1,repr_size)
        token_attn_value = torch.bmm(token_attn_param, token_rnn_output).squeeze(1)  # (batch_size,max_len)
        token_attn_value = F.softmax(token_attn_value.masked_fill(token_mask == 0, -1e10), dim=1)  # 掩码 + softmax  (batch_size,max_len)

        token_rnn_output = token_rnn_output.permute(0, 2, 1)  # (batch_size,max_len,rep_size
        token_attn_value = token_attn_value.unsqueeze(1)  # (batch_size,1,max_len)
        token_repr = torch.bmm(token_attn_value, token_rnn_output).squeeze(1)  # (batch_size,rep_size)

        # 处理ast
        ast_rnn_output = self.tree_linear(self.tree_encoder(asts)).permute(0, 2, 1)
        ast_attn_param = self.tree_attn_param.repeat(batch_size, 1).unsqueeze(1)
        ast_attn_value = torch.bmm(ast_attn_param, ast_rnn_output).squeeze(1)
        ast_attn_value = F.softmax(ast_attn_value.masked_fill(ast_mask == 0, -1e10), dim=1)

        ast_rnn_output = ast_rnn_output.permute(0, 2, 1)  # (batch_size,max_len,rep_size
        ast_attn_value = ast_attn_value.unsqueeze(1)  # (batch_size,1,max_len)
        ast_repr = torch.bmm(ast_attn_value, ast_rnn_output).squeeze(1)  # (batch_size,rep_size)

        # 结合token以及ast向量
        code_repr = self.combine_linear(torch.cat([token_repr, ast_repr], dim=1))

        return code_repr

    # 对description进行编码
    def desc_encode(self, desc, desc_len):
        batch_size = desc.shape[0]
        desc_mask=self.create_mask(desc_len)
        desc_rnn_output = self.desc_linear(self.desc_encoder(desc, desc_len)).permute(0, 2, 1)  # (batch_size, repr_size, max_len)
        desc_attn_param = self.desc_attn_param.repeat(batch_size, 1).unsqueeze(1)  # (batch_size,1,repr_size)
        desc_attn_value = torch.bmm(desc_attn_param, desc_rnn_output).squeeze(1)  # (batch_size,max_len)
        desc_attn_value = F.softmax(desc_attn_value.masked_fill(desc_mask == 0, -1e10), dim=1)  # mask + softmax  (batch_size,max_len)

        desc_rnn_output = desc_rnn_output.permute(0, 2, 1)  # (batch_size,max_len,rep_size
        desc_attn_value = desc_attn_value.unsqueeze(1)  # (batch_size,1,max_len)
        desc_repr = torch.bmm(desc_attn_value, desc_rnn_output).squeeze(1)  # (batch_size,rep_size)

        return desc_repr

    def similarity(self, code_vec, desc_vec):
        """
        https://arxiv.org/pdf/1508.01585.pdf 
        """
        assert self.sim_measure in ['cos', 'poly', 'euc', 'sigmoid', 'gesd', 'aesd'], "invalid similarity measure"
        if self.sim_measure == 'cos':
            return F.cosine_similarity(code_vec, desc_vec)
        elif self.sim_measure == 'poly':
            return (0.5*torch.matmul(code_vec, desc_vec.t()).diag()+1)**2
        elif self.sim_measure == 'sigmoid':
            return torch.tanh(torch.matmul(code_vec, desc_vec.t()).diag()+1)
        elif self.sim_measure in ['euc', 'gesd', 'aesd']:
            euc_dist = torch.dist(code_vec, desc_vec, 2)  # or torch.norm(code_vec-desc_vec,2)
            euc_sim = 1 / (1 + euc_dist)
            if self.sim_measure == 'euc':
                return euc_sim
            sigmoid_sim = torch.sigmoid(torch.matmul(code_vec, desc_vec.t()).diag()+1)
            if self.sim_measure == 'gesd':
                return euc_sim * sigmoid_sim
            elif self.sim_measure == 'aesd':
                return 0.5*(euc_sim+sigmoid_sim)

    def create_mask(self,src_lens):
        max_len = max(src_lens)
        mask = []
        mask = np.array(mask)

        for i in range(len(src_lens)):
            row = np.concatenate(
                (np.ones(src_lens[i]), np.zeros(max_len - src_lens[i])))
            mask = np.concatenate((mask, row))

        mask = mask.reshape(len(src_lens), max_len)

        return torch.from_numpy(mask).cuda()

    def forward(self, tokens, token_len, ast, ast_len, desc_anchor, desc_anchor_len, desc_neg, desc_neg_len):

        # 计算各部分的向量表示
        code_repr = self.code_encode(tokens, token_len, ast, ast_len)
        desc_anchor_repr = self.desc_encode(desc_anchor, desc_anchor_len)
        desc_neg_repr = self.desc_encode(desc_neg, desc_neg_len)

        # 计算相似度
        anchor_sim = self.similarity(code_repr, desc_anchor_repr)
        neg_sim = self.similarity(code_repr, desc_neg_repr)  # [batch_sz x 1]

        loss = (self.margin-anchor_sim+neg_sim).clamp(min=1e-6).mean()
        return loss
