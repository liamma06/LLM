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
                print(f"Ep: {epoch+1} (Step {global_step:06d}): " f"Train loss {train_loss: .3f}")

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