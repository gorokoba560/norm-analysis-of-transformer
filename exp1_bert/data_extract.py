import pickle
import torch
from tqdm import tqdm
from transformers import BertTokenizer


path = './data/unlabeled_attn.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

input_ids_lis = []
seq_len_lis = []
token_type_ids_lis = []
attention_mask_lis = []

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sep_ids = tokenizer.convert_tokens_to_ids("[SEP]")

for dic in tqdm(data):
    tokens = dic["tokens"]
    seq_len = len(tokens)
    seq_len_lis.append(seq_len)
    input_ids, attention_mask = tokenizer.encode_plus(tokens, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=True, padding='max_length', max_length=128).values()
    input_ids_lis.append(input_ids)
    attention_mask_lis.append(attention_mask)
    first_sep_pos = input_ids.index(sep_ids)
    token_type_ids_lis.append([0]*(first_sep_pos+1) + [1]*(128-first_sep_pos-1))

input_ids = torch.tensor(input_ids_lis)
attention_mask = torch.tensor(attention_mask_lis)
token_type_ids = torch.tensor(token_type_ids_lis)
seq_len = torch.tensor(seq_len_lis)

torch.save(seq_len, './work/seq_len.pt')
torch.save(input_ids, './work/input_ids.pt')
torch.save(attention_mask, './work/attention_mask.pt')
torch.save(token_type_ids, './work/token_type_ids.pt')