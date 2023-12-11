import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import os
from tqdm import tqdm, trange

# File That Reads in and Processes the cover-letter-dataset

class CoverLetterDataset(Dataset):
    def __init__(self, df):
        super().__init__()

        self.data_pairs = []    
        for itr, row in  df.iterrows():
            prompt = row['Prompt']
            letter = row['Cover Letter']
            data = prompt + letter + "<|endoftext|>"

            self.data_pairs.append(data)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, item):
        return self.data_pairs[item]


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

def build_prompt(work_experience, qualifications, company, job_title, skills, applicant):
    prompt = ''
    prompt += f' <|workexperience|> {work_experience} <|endofworkexperience|>'
    prompt += f' <|qualifications|> {qualifications} <|endofqualifications|>'
    prompt += f' <|company|> {company} <|endofcompany|>'
    prompt += f' <|jobtitle|> {job_title} <|endofjobtitle|>'
    prompt += f' <|skills|> {skills} <|endofskills|>'
    prompt += f' <|applicant|> {applicant} <|endofapplicant|>'

    #prompt += f' Dear Hiring Manager,\n' #Not sure if we want this \n here

    return prompt

def build_prompt_from_row(row):
    prompt = build_prompt(row['Current Working Experience'],
            row['Preferred Qualifications'],
            row['Hiring Company'],
            row['Job Title'],
            row['Skillsets'],
            row['Applicant Name'] )

    return prompt

def add_prompts(df):
    df['Prompt'] = df.apply(lambda x: build_prompt_from_row(x), axis=1)
    return df


if __name__ == '__main__':
    df = get_data('data/cover-letter-dataset', 'train.csv')
    df = add_prompts(df)
    dataset = CoverLetterDataset(df)

