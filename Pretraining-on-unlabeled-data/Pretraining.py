import torch 
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPT_model.GPT_Model_architecture_class import generate_text_simple

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
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs = 10, eval_freq = 1, eval_iter = 1, start_context = "", tokenizer = None):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen , global_step = 0 , -1

    for epoch in range(num_epochs):
        model.train()

        for input_bath, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_bath, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_bath.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, vall_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(vall_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep: {epoch+1} (Step {global_step:06d}): " f"Train loss {train_loss: .3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter )

    model.train()
    return train_loss, val_loss

def generate_and_print_sample (model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx = encoded, max_new_tokens = 50, context_size = context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()