import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import os


#cover-letter-dataset

class CoverLetter(Dataset):  
    def __init__(self, df, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.letter = []

        for row in df['Cover Letter']:
          self.letter.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
            ))               
        if truncate:
            self.letter = self.letter[:20000]
        self.letter_count = len(self.letter)
        
    def __len__(self):
        return self.letter_count

    def __getitem__(self, item):
        return self.letter[item]
    

def get_data(data_dir, data_file):
    path = os.path.join(data_dir, data_file)
    df = pd.read_csv(path)
    df = df[df['Cover Letter'].apply(lambda x: len(x.split(' ')) < 350)] #Filtering ones that are too long? Dont think this does anything
    return df

def split_johns(df):
    '''
     Letters from John Smith seem to be structured differently
    '''
    return df[df['Applicant Name'] == 'John Smith'], df[df['Applicant Name'] != 'John Smith']

if __name__ == '__main__':
    train = get_data('data/cover-letter-dataset', 'train.csv')
    dataset = CoverLetter(train, train['Cover Letter'], truncate=True, gpt2_type="gpt2")



