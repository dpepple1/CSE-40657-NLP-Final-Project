import sys
import torch
import gpt
import os
import gpt as gpt
import cover_letter_datareader as cld
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup


def readfile(file):
    with open(file, 'r') as fh:
        lines = fh.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].rstrip()
            #lines[i] = lines[i].split(' ') #Not sure if we need this
        return lines
    raise Exception('Error Opening File:', file)

def greedy(p):
    a= torch.argmax(p)
    return a

def ancestral(p):
    #return torch.randint(0, p.size(dim=0), (1,))
    new_p = torch.exp(p)
    return torch.multinomial(new_p, 1)

def top_p(probs, p):
    new_probs = torch.exp(probs)
    sorted_probs, indicies = torch.sort(new_probs, dim=0, descending = True)

    cumsums = torch.cumsum(sorted_probs, dim=0)
    end = 0
    for i in range(cumsums.size(dim=0)):
        if cumsums[i] >= p:
            end = i + 1
            break

    dist_ind = indicies[0:end]
    dist = torch.index_select(new_probs, 0, dist_ind)
    
    sample = torch.multinomial(dist, 1)
    return indicies[sample]

def train(dataset, test, lm, batch_size=16, epochs=5, lr=2e-5,
            max_seq_len=400, warmup_steps=200, gpt2_type="gpt2", 
            output_dir=".", output_prefix="wreckgar", test_mode=False,
            save_model_on_epoch=False):

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda")
    
    lm.model = lm.model.cuda()

    lm.model.train()
    optimizer = AdamW(lm.model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)
    proc_seq_count = 0
    loss = 0.0
    batch_count = 0


    tmp_prompt_tens = None
    models_folder = "trained_models"
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    input_tensor = None

    for epoch in range(epochs):
        print(f"Training Epoch {epoch}")
        print(loss)
        for idx, text in enumerate(tqdm(loader)):
            
            tokenized = torch.tensor(lm.tokenize(text))
            #tokenized = self.tokenizer.encode(text)
            tokenized = tokenized.to(device)
            #print(type(tokenized), tokenized)

            (input_tensor, carry_on, remainder) = lm.pack_tensor(tokenized, input_tensor, 768) #arbitraty 768
            if carry_on and idx != len(loader) - 1:
                continue

            input_tensor = input_tensor.to(device)

            outputs = lm.model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                lm.model.zero_grad()
            
            batch_count += 1 
            input_tensor = None

        if save_model_on_epoch:
            torch.save(
                lm.model.state_dict(),
                os.path.join('trained_models', f'trained-gpt2--{epoch}.pt')
            )

        lm.model = lm.model.to('cpu')
        validate(lm, epoch, test)
        lm.model = lm.model.cuda()

    
            #keep reading here: https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7


def validate(lm, epoch, df):
    output_folder = "sample_output"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for idx, row in df.iterrows():

        if idx > 5:
            break
            
        prompt = row['Prompt']
        prompt += ' Dear Hiring Manager, \n'
        
        q = lm.start()
        prev = lm.bos()
        total = 0.

        nums = lm.tokenize(prompt)
        assert prompt == lm.detokenize(nums)

        for num in nums:
            q, p = lm.step(q, prev)
            prev = num

        output_len = 0
        output = []

        while output_len < 200:
            q, p = lm.step(q, prev)
            #selected = greedy(p)
            #selected = ancestral(p)
            selected = top_p(p, 0.5)
            prev = selected
            output.append(lm.detokenize(selected))
            output_len += 1

        output = ''.join(output)

        folder_path = os.path.join(output_folder, f'epoch{epoch}')
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        file_path = os.path.join(folder_path, f'test_output{idx}.txt')
        with open(file_path, 'w') as fh:
            fh.write(prompt + output + '\n')

if __name__ == "__main__":

    df = cld.get_data('data/cover-letter-dataset', 'train.csv')
    df = cld.add_prompts(df)
    #johns, df = cld.split_johns(df)
    dataset = cld.CoverLetterDataset(df)

    lm = gpt.LanguageModel('gpt2', 'cpu')

    test = cld.get_data('data/cover-letter-dataset', 'test.csv')
    test = cld.add_prompts(test)

    train(dataset, test, lm, epochs=20, save_model_on_epoch=True)


#IDEAS:
# ADD <|endoftext|> to the end of the training data.