"""
The purpose of this file is to continue testing on already trained models.
"""

import sys
import torch
import gpt
import os
import gpt as gpt
import cover_letter_datareader as cld

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


lm = gpt.LanguageModel('gpt2', 'cpu')
lm.model.load_state_dict(torch.load('trained_models/trained-gpt2--10.pt'))

df = cld.get_data('data/cover-letter-dataset', 'test.csv')
df = cld.add_prompts(df)

output_folder = "sample_output"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

subset = df.sample(frac=1)


for idx, (_, row) in enumerate(subset.iterrows()):

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

    while output_len < 1000:
        q, p = lm.step(q, prev)
        #selected = greedy(p)
        #selected = ancestral(p)
        selected = top_p(p, 0.6)
        prev = selected
        word = lm.detokenize(selected)
        if word == '<|endoftext|>':
            break
        output.append(word)
        output_len += 1

    output = ''.join(output)

    folder_path = os.path.join(output_folder, 'testing')
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    file_path = os.path.join(folder_path, f'test_output{idx}.txt')
    with open(file_path, 'w') as fh:
        fh.write(prompt + '\n<START>\n'+output + '\n')