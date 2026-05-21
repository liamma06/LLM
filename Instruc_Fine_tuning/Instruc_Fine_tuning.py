#defining a format for input data for instruction fine-tuning
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

#------------------------#

import torch 
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer): 
        self.data = data
        self.encoded_texts = []

        #iterate through the data and properly format 
        for entry in self.data:
            instruction_plus_input = format_input(entry)

            #include the response
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text

            #tokenize the full text and store the encoded version
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    
    def __getitem__(self,index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

#without target tensor, Only inputs tensor
def custom_collate_draft_1(batch, pad_token_id = 50256, device="cpu"):

    #finding maximum length of sequence in batch
    batch_max_length = max(len(item)+1 for item in batch)

    inputs_lst = []

    #iterate through batch
    for item in batch:
        new_item = item.copy() #to avoid modifying original item in batch
        new_item += [pad_token_id] #add pad token at the end of the sequence

        #pad the sequence to the maximum length in batch
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        #convert to tensor and remove the last token (the pad token we just added) for input
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    #stack the input tensors and move to device
    inputs_tensor = torch.stack(inputs_lst).to(device)

    return inputs_tensor

#generate target tensor and inputs tensor 
def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

#generate target tensor and inputs tensor with -100 paddings 
def custom_collate_fn(batch, pad_token_id = 50256, ignore_index= -100, allowed_max_length=None, device="cpu"):
    
    #finding maximum length of sequence in batch
    batch_max_length = max(len(item)+1 for item in batch)

    inputs_lst, targets_lst = [], []

    #iterate through batch
    for item in batch:
        new_item = item.copy() #to avoid modifying original item in batch
        new_item += [pad_token_id] #add pad token at the end of the sequence

        #pad the sequence to the maximum length in batch
        padded = (
            new_item + [pad_token_id] * (batch_max_length - len(new_item))
        )
        #convert to tensor and remove the last token (the pad token we just added) for input
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:]) #targets are the same sequence but shifted by one token to the right
        
        #create list of True and False values. Marks True anywhere target tensor has a pad token
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        #skip first padding and rest make them the ignore_index
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        #if allowed_max_length is set, truncate the inputs and targets to that length
        #not needed
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        
        
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    #stack the input tensors and move to device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor

 

