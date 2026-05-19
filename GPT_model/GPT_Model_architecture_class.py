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
    
#GELU activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))

#feed forward network module 
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            #expand the embedding dimension to 4 times its size, then apply GELU activation, then project back down to original embedding dimension
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )
    def forward(self, x):
        return self.layers(x)

#Example Deep Neural Network to show shortcut connections 
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut

        #create a list of layers based on the provided layer sizes (input, output), each followed by a GELU activation
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        #iterative function to compute x for every layer 
        for layer in self.layers:
            layer_output = layer(x) #compute the output of the current layer, input into layer

            #if use_shortcut is true and the input and output shapes match, add the input to the output (residual connection)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

def print_gradients(model,x):
    output = model(x)#runs forward pass to compute output
    target = torch.tensor([[0.]]) #dummy target
    
    loss = nn.MSELoss()
    loss = loss(output, target)#compute loss between model output and target

    loss.backward() #computes gradients of loss with respect to all model parameters, Stored in .grad

    for name, param in model.named_parameters():
        if 'weight' in name:

            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


#Transfomer Block
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention.MultiHead_attention_class import MultiHeadAttention
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['drop_rate'],
            num_heads=cfg['num_heads'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):

        #stage 1: attention with shortcut connection
        shortcut = x
        x = self.norm1(x) #stablize
        x = self.att(x) #apply attention
        x = self.drop_shortcut(x) #random dropout for regularization
        x = x + shortcut #blend original with new

        #stage 2: feed forward network with shortcut connection
        shortcut = x 
        x = self.norm2(x)
        x = self.ff(x) #apply feed forward network
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
    
# GPT MODEL ARCHITECTURE IMPLEMENTATION 
class GPTModel (nn.Module):
    def __init__(self,cfg):
        super().__init__()
        #token embedding layer to convert input token to embeddings
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        #stack of transformer blocks
        #nn.Sequential allows us to stack multiple layers together, and it will pass the input through each layer in order
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['num_layers'])]
        )

        #final layer norm before output projection, helps stabilize training and improve performance
        self.final_norm = LayerNorm(cfg['emb_dim'])

        #output head to project the final embeddings to the vocabulary size, allowing us to get logits for each token in the vocabulary (for next token prediction)
        #bias is false because we want the model to learn to predict the next token based solely on the input embeddings, without any additional bias term
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx) #convert input token indices to embeddings
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device) #create position indices for the sequence and convert to embeddings
        )
        x = tok_embeds + pos_embeds #final embedding with both token and position
        x = self.drop_emb(x) #dropout 
        x = self.trf_blocks(x) #pass data through stack of transformer block
        x = self.final_norm(x) #final layer norm for stablization
        logits = self.out_head(x) #project to vocab size for logits (prediction of next token)
        return logits

#simple generate text function 
def generate_text_simple( model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        #only take the last context size as inputs to model
        #this is because the model can only attend to a limited context size, so we need to make sure we are only feeding in the most recent tokens that fit within that context window
        idx_cond = idx[:, -context_size:]


        with torch.no_grad():# tells pytorch not to computes gradients (we aren't training)
            logits = model(idx_cond)

        #take very last token (keeps batch same ,and all scores across vocab)
        logits = logits[:, -1, :]

        #apply softmax to turn into probabilites scores
        probas = torch.softmax(logits, dim=-1)

        #take token with highest probability as the next token, 
        #keepdim true to maintain the same number of dimensions (for concatenation later)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        #generate the next token and append to existing sequence of tokens 
        idx = torch.cat((idx, idx_next), dim=1)

    return idx