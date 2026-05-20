
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