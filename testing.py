import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv

### Prepare data
df = pd.read_csv('data/cover-letter-dataset/train.csv')
#lyrics = lyrics[lyrics['Idiom']=='ENGLISH']

#Only keep popular artists, with genre Rock/Pop and popularity high enough
#artists = pd.read_csv('artists-data.csv')
#artists = artists[(artists['Genre'].isin(['Rock'])) & (artists['Popularity']>5)]
#df = lyrics.merge(artists[['Artist', 'Genre', 'Link']], left_on='ALink', right_on='Link', how='inner')
#df = df.drop(columns=['ALink','SLink','Idiom','Link'])

#Drop the songs with lyrics too long (after more than 1024 tokens, does not work)
df = df[df['Cover Letter'].apply(lambda x: len(x.split(' ')) < 350)]

#Create a very small test set to compare generated text with the reality
test_set = df.sample(n = 200)
df = df.loc[~df.index.isin(test_set.index)]

#Reset the indexes
test_set = test_set.reset_index()
df = df.reset_index()

#For the test set only, keep last 20 words in a new column, then remove them from original column
test_set['True_end_lyrics'] = test_set['Cover Letter'].str.split().str[-20:].apply(' '.join)
test_set['Cover Letter'] = test_set['Cover Letter'].str.split().str[:-20].apply(' '.join)

class SongLyrics(Dataset):  
    def __init__(self, control_code, truncate=False, gpt2_type="gpt2", max_length=1024):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.lyrics = []

        for row in df['Cover Letter']:
          print(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
          self.lyrics.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
            ))               
        if truncate:
            self.lyrics = self.lyrics[:20000]
        self.lyrics_count = len(self.lyrics)
        
    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return self.lyrics[item]
    
dataset = SongLyrics(df['Cover Letter'], truncate=True, gpt2_type="gpt2")    