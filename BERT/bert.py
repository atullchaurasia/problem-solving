import torch
import torch.nn as nn

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size,
                 n_segments,
                 max_len,
                 embed_dim,
                 dropout):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.seg_embed = nn.Embedding(n_segments, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.pos_inp = torch.tensor([i for i in range(max_len)], )
   
    def forward(self, seq, seg):     
        embed_val1 = self.tok_embed(seq) + self.seg_embed(seg) + self.pos_embed(self.pos_inp)
        return embed_val1
    
class Bert(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 n_segments,
                 max_len, 
                 embed_dim,
                 n_layers,
                 attn_heads,
                 dropout):
        super().__init__()
        self.embedding = BertEmbedding(vocab_size, n_segments, max_len, embed_dim, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(embed_dim, attn_heads, embed_dim*4)
        self.encoder_block = nn.TransformerEncoder(self.encoder_layer, n_layers)
        
    def forward(self, seq, seg):
        out = self.embedding(seq, seg)
        out = self.encoder_block(out)
        return out    
    
    

if __name__ == '__main__':
    vocab_size:int = 30000 
    n_segments:int = 3
    max_len:int = 512
    embed_dim:int = 768
    n_layers:int = 12
    attn_heads:int = 12
    dropout:float = 0.1
    
    sample_seq = torch.randint(high=vocab_size, size=[max_len, ])
    sample_seg = torch.randint(high=n_segments, size = [max_len, ])
    
    embedding = BertEmbedding(vocab_size, n_segments, max_len, embed_dim, dropout)
    embedding_tensor = embedding(sample_seq, sample_seg)
    print(embedding_tensor.size())
    
    
    bert = Bert(vocab_size, n_segments, max_len, embed_dim, n_layers, attn_heads, dropout)
    out = bert(sample_seq, sample_seg)
    print(out.size())
