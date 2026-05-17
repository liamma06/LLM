import torch
import torch.nn as nn

class DummyGPTmodel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
    
        #placeholder for transformer block
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg['num_layers'])]
        )

        #placeholder for layer norm 
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        #embedds for the input tokens 
        tok_embeds = self.tok_emb(in_idx)

        #embedds for the position of the tokens in the sequence
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        #combine token and position embeddings, then apply dropout
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        #pass to transformers and final norm, placeholder for now 
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        #project to vocab size for logits (prediction of next token)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self, x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    
    def forward(self, x):
        return x
    
# normalization layer
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 #small constant to prevent divison by zero

        #scale and shift are parameters that allow the model adjust/ optimize
        #sometimes normalizing to zero mean and unit variance can limit the model's ability to learn complex patterns,
        #so we allow it to learn how much to scale and shift the normalized output
        #Summary: give back some freedom to the model
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps) #normalize to zero mean and unit variance

    
        return self.scale * norm_x + self.shift