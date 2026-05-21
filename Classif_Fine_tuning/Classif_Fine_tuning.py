
import pandas as pd

#split dataset into train, validation and test sets
def random_split(df, train_frac, validation_frac):

    #shuffle dataset 
    df = df.sample(
        frac=1,#all rows 
        random_state = 123# seed
    ).reset_index(drop=True) #reset index after shuffling

    train_end= int(train_frac * len(df)) #index where training set ends(int to remove decimal)
    validation_end = train_end + int(validation_frac * len(df)) #index where validation set ends

    train_df = df[:train_end] 
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df

#Dataloader for spam dataset using paddings
import torch 
from torch.utils.data import Dataset 
from torch import device 
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length = None, pad_token_id = 50256):
        self.data = pd.read_csv(csv_file)

        #encode the text data using the provided tokenizer
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        #if max_length not provided calculate longest encoded text 
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else: 
            self.max_length = max_length
            #if text longer than provided max_length onyl keep tokens up to max_length (truncate)
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]
        #Fills the remaining space with the pad_token_id until max_length is reached (padding)
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data["Label"].iloc[index] #go to labels and get label at index
        return (
            torch.tensor(encoded, dtype=torch.long), #convert list of encoded tokens to tensor of type long
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)
    
    #basic function to calculate longest by going thorugh each token and checking
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

#---------------------------------------#

#Classification accuracy 
def calc_accuracy_loader(data_loader, model, device, num_batches = None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    #determine how many batches to evalute on 
    if num_batches is None: 
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    #iterate through batches of data and make predictions using the model, then compare predicted labels to true labels to calculate accuracy
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:,-1,:] #get logits only for last token in sequence (for classification)

            predicted_labels = torch.argmax(logits, dim=-1)#take index of highest logit 

            num_examples += predicted_labels.shape[0]#number of examples is number of predicted labels (or target labels) in batch
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )  
        else:
            break
    return correct_predictions / num_examples   

#calculate loss for a single batch of input and target data
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
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

#---------------------------------------#
#fine-tuning loop for classification task
def train_classifier_simple (model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    #iterate through epochs 
    for epoch in range(num_epochs):
        model.train()

        #iterate through batches of training data 
        for input_batch, target_batch in train_loader:
            #training loop 
            optimizer.zero_grad() 
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            #evaluate model on training and validation data at specified frequency (eval_freq) and number of batches (eval_iter)
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        #after each epoch evaluate accuracy on training and validation data        
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")

        #append accuracy values to lists for tracking
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen

#evaluate model on training and validation data by calculating loss using calc_loss_loader for specified number of batches (eval_iter)
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
        
#---------------------------------------#
#funciton to classify text as spam or not 
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # tokenize imputs and prepare for model input
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    
    # Pad sequences to the longest sequence
    #though technically we don't need as we had to do it for our dataset
    #but we want to math the conditions of the training
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"