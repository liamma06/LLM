import torch
import sys, os

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from GPT_model.GPT_Model_architecture_class import generate_text_simple
except ImportError:
    from GPT_Model_architecture_class import generate_text_simple

#convert text to token ids and back using the tokenizer.
def text_to_token_ids (text, tokenizer):
    encoded = tokenizer.encode (text, allowed_special = {'<|endoftext|>'})

    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension

    return encoded_tensor

def token_ids_to_text (token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    text = tokenizer.decode(flat.tolist()) #convert tensor to list of ints 
    return text



#calculate loss for a single batch of input and target data
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten(0,1)
    )
    return loss

#iterate through training batches and calc_loss_batch on each, then global avg loss
def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0.0
    
    #set limit on how many batches to process. 
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    #go through the batches 
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches #average loss across batches


#main function for pretraining LLMs 
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    
    #global step keep track of how many batches we've processed
    #tokens_seen keeps track of how many tokens we've processed across all batches, which is useful for monitoring training progress and scheduling evaluations.
    tokens_seen , global_step = 0 , -1

    #traing for number of epochs
    for epoch in range(num_epochs):
        model.train()

        #iterate through data loader
        for input_bath, target_batch in train_loader:
            optimizer.zero_grad() #remove old gradient from previous batch
            loss = calc_loss_batch(input_bath, target_batch, model, device) #calculate loss for current batch (forward pass)
            loss.backward() #compute gradients (backward pass -> loss for every weight param)
            optimizer.step() #update model weights based on computed gradients (gradient descent step)
            tokens_seen += input_bath.numel() #num of elements -> ex [2,256] =512 tokens seen
            global_step += 1

            #evaluate model on train and vali every eval_freq steps and store losses and tokens seen for monitoring training progress.
            if global_step % eval_freq == 0:
                train_loss, vall_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(vall_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep: {epoch+1} (Step {global_step:06d}): " f"Train loss {train_loss: .3f}" f"Val loss {val_loss: .3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

#evaluate model 
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter )

    model.train()
    return train_loss, val_loss

#generate text sample from model 
def generate_and_print_sample (model, tokenizer, device, start_context):
    model.eval()
    
    #read dimension of model positional embedding layer and tells the max number of token rows it can accept
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device) #convert start context to tokens and input tensor
    
    with torch.no_grad():
        #generate text by feeding the model the encoded start context and letting it predict the next tokens until it reaches the max_new_tokens limit or the end of text token.
        #by passing model it uses the model's weights to predict
        token_ids = generate_text_simple(
            model=model, idx = encoded, max_new_tokens = 50, context_size = context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)#convert generated token ids back to text
    print(decoded_text.replace("\n", " "))
    model.train()


#new generation with temperature sampling and top K
def generate(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx [:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1,:] #take last token logits for next token prediction

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k) #get top K logits 
            min_val = top_logits[:,-1] #lower bound for logits
            logits = torch.where(
                logits < min_val, #condition
                torch.tensor(float("-inf"), device=logits.device), #set -inf to exclude
                logits #rest stay same
            )

        if temperature > 0.0:
            logits = logits / temperature #scale by temperature

            #subtract max logit for numerical stability before applying softmax (prevents overflow)
            logits = logits - logits.max(dim=-1, keepdim=True).values 

            probs = torch.softmax(logits, dim=-1) #softmax to get probabilities

            #choose next token by sampling from the probability distribution defined by the logits (after applying temperature and top-k filtering)
            idx_next = torch.multinomial(probs, num_samples=1) 
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break
            
        #append prediction to existing sequence of tokens
        #needed to be included in next input for next word. 
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

####LOADING WEIGHTS
import numpy as np

#function to load the pretrained weights into our GPT model
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))



#inject wieghts into our custom pytorch model
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
        

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])