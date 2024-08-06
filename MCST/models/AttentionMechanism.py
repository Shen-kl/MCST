import torch.nn as nn
import torch
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        # embed_size:表示将embedding，即词向量的维度
        # heads: 设置多头注意力的数量。如果设置为 1，那么只使用一组注意力。如果设置为其他数值，那么 heads 的值需要能够被 embed_size 整除
        super(SelfAttention,self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size),"Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.keys = nn.Linear(self.head_dim,self.head_dim,bias=True)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.fc_out = nn.Linear(heads*self.head_dim,embed_size)

    def  forward(self, values, keys, query, mask):
        N = query.shape[0]  # N 表示 batch_size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # query_len是词向量的个数 这里代表目标的个数
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim) # 分成多头 分别进行self-attention
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk",[queries, keys])
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape :(N, heads, query_len,  key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)  # 在第key维度进行softmax操作

        out = torch.einsum("nhql,nlhd->nqhd",[attention, values]).reshape( # key_len 等于 value_len
            N, query_len, self.heads * self.head_dim # 将多头拼接在一起 之后再去做全连接
        )
        # attention shape: (N, heads, query_len,  key_len)
        # values shape : (N, value_len, heads, heads_dim)
        # after einsum (N, query_len, heads ,heads_dim) then flatten last two dimensions

        out = self.fc_out(out)
        return out

class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = torch.softmax(scores,dim=-1)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)
