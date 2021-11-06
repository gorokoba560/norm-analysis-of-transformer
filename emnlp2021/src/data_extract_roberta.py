from transformers import BertTokenizer, RobertaTokenizer
import pickle
import torch
from tqdm import tqdm
import os


bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

path = './data/unlabeled_attn.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    
sent_pair_lis = []

for dic in tqdm(data):
    tokens = dic["tokens"]
    ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    sent = bert_tokenizer.decode(ids)
    sent1, sent2 = sent[5:-5].strip().split("[SEP]")
    sent_pair_lis.append([sent1, sent2])
    
input_ids, attention_mask = roberta_tokenizer(
    sent_pair_lis, 
    padding="longest", 
    return_tensors="pt"
).values()

if not os.path.exists(f"./work/roberta_data"):
    os.makedirs(f"./work/roberta_data")

torch.save(input_ids, './work/roberta_data/input_ids.pt')
torch.save(attention_mask, './work/roberta_data/attention_mask.pt')