import torch
import torch.nn as nn
from causal_attention_class import CausalAttention

class MultiHeadAttentionWrapper(nn.Module):
    def __init__ (self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        #special container tells pytorch every object in list is a sub module. Alloing the wieght matricies of each head to be registered as parameters and updated during training
        self.heads = nn.ModuleList(
            #creates independent instances for number of heads
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)]
        )

    def forward(self, x):
        #loops through each head and applies it to the input x, then concatenates the outputs along the last dimension (the feature dimension) to combine the outputs of all heads into a single tensor
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        #ensures that each head will have the same number of tokens in the output.
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads #dimension of each head. 
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        #reshapes k,q,v to separate heads 
        #b = batch size (does not change)
        #num_tokens = number of tokens in input sequence (does not change)
        #self.num_heads = number of attention heads (new dimension for heads)
        #self.head_dim = dimension of each head (new dimension for head features)
        #this part confused me a lot but is it easier to think of it essientally grouping the features into num_heads groups, each of size head_dim. So instead of having one big feature dimension of size d_out, we have num_heads smaller feature dimensions of size head_dim, one for each head. This allows us to perform attention separately for each head and then combine the results later.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        #swap token and head needed for the attneion scores calculation (like every row(token) has the weights accross all heads for that token)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        #dotting the words and data togther while ignoreing the rest
        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vecs = (attn_weights @ values).transpose(1,2)

        context_vecs = context_vecs. contiguous().view(b, num_tokens, self.d_out)
        context_vecs = self.out_proj(context_vecs)
        return context_vecs