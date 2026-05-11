import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        #initialize dropout layer
        self.dropout = nn.Dropout(dropout)

        #register buffer, buffers are automatically moved to the same device(CPU, GPU) as the model and are not updated during training
        self.register_buffer(
            "mask",
            #create upper triangle with 1s above the diagonal and 0s on and below the diagonal
            torch.triu(torch.ones(context_length, context_length), 
                        diagonal=1)
        )

    def forward(self, x):
        #b -> batch size(how many sentences or like groups
        #num_tokens -> how many tokens in the context(each batch)
        #d_in -> dimension of the input embeddings
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        #when we calc attn scores we need to use .T(transpose) to get the correct shape for matrix multiplication(2D)
        #however we currently have 3D tensors (batch, Token, embedding/features) so we need to transpose the last two dimensions to get (batch, embedding/features, Token) 
        #this way we can do matrix multiplication for all sentences in the batch at once
        attn_scores = queries @ keys.transpose(1,2)

        #mask the attn scores to prevent attention to future tokens, we use masked_fill to replace the masked positions with -inf so that after softmax they become 0
        attn_scores.masked_fill_( #masked_fill_ is basically a find and replace function if true do this if false do nothing
            #call the mask variable from the buffer and convert to bool
            self.mask.bool()[:num_tokens, :num_tokens], 
            -torch.inf
        )

        attn_weights = torch.softmax(attn_scores / d_in**0.5, dim=-1)
        
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values

        return context_vec