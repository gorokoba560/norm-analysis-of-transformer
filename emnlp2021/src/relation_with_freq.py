import os
from tqdm import tqdm
import random
import argparse
import torch
from transformers import BertTokenizer, BertModel


device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="base")
args = parser.parse_args()

if args.model == "base":
    pretrained = "bert-base-uncased"
    layer_num = 12
elif args.model == "large":
    pretrained = "bert-large-uncased"
    layer_num = 24
elif args.model == "medium":
    pretrained = "prajjwal1/bert-medium"
    layer_num = 8
elif args.model == "small":
    pretrained = "prajjwal1/bert-small"
    layer_num = 4
elif args.model == "mini":
    pretrained = "prajjwal1/bert-mini"
    layer_num = 4
elif args.model == "tiny":
    pretrained = "prajjwal1/bert-tiny"
    layer_num = 2
else:
    print("args 'model' should be 'base', 'large', 'medium', 'small', 'mini', or 'tiny'.")
    exit()

# load preprocessed BERT inputs (Wikipedia data)
input_ids = torch.load("./work/input_ids.pt")
attention_mask = torch.load('./work/attention_mask.pt')
token_type_ids = torch.load('./work/token_type_ids.pt')
seq_len = torch.load('./work/seq_len.pt')

# make rank matrix & count matrix for inputs (Wikipedia data)
input_rank = input_ids.clone()
input_memo = input_ids.clone()

# rank_count_ids was created by counting the BERT training dataset we reproduced
with open ("./data/rank_count_ids.txt") as f:
    for line in f:
        rank, count, ids = map(int, line.split())
        input_rank[input_ids==ids] = rank
        input_memo[input_ids==ids] = -1

# assign -1 to pad
input_rank[input_memo==0] = -1

# assign bottom rank to tokens which doesn't have count/rank
input_rank[input_memo>0] = rank + 1

rank = input_rank[input_ids!=0]
if not os.path.exists(f"./work/relation_with_freq"):
    os.makedirs(f"./work/relation_with_freq")
torch.save(rank, './work/relation_with_freq/rank.pt')

model = BertModel.from_pretrained(pretrained).to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained(pretrained)

attn_w_ratio_lis =  [[] for _ in range(layer_num)]
attn_n_ratio_lis =  [[] for _ in range(layer_num)]
attnres_w_ratio_lis =  [[] for _ in range(layer_num)]
attnres_n_ratio_lis =  [[] for _ in range(layer_num)]
attnresln_n_ratio_lis =  [[] for _ in range(layer_num)]

with torch.no_grad():
    for i in tqdm(range(len(input_ids))):
        ids = torch.unsqueeze(input_ids[i], 0).to(device)
        mask = torch.unsqueeze(attention_mask[i], 0).to(device)
        type_ids = torch.unsqueeze(token_type_ids[i], 0).to(device)
        _, _, attentions, norms = model(ids, token_type_ids=type_ids, attention_mask=mask, output_attentions=True, output_norms=True)
        
        for l in range(layer_num):
            head_num = len(attentions[l][0])
            avg_attn = torch.sum(attentions[l][0], dim=0) / head_num

            # Attn-W
            ratios = (torch.sum(avg_attn, dim=1) - torch.diagonal(avg_attn))
            attn_w_ratio = ratios[ids[0]>0]
            attn_w_ratio_lis[l].append(attn_w_ratio)

            # Attn-N
            attn_n_ratio = norms[l][-3][ids>0].cpu()
            attn_n_ratio_lis[l].append(attn_n_ratio)

            # AttnRes-W
            ratios = (torch.sum(avg_attn, dim=1) - torch.diagonal(avg_attn)) / 2
            attnres_w_ratio = ratios[ids[0]>0]
            attnres_w_ratio_lis[l].append(attnres_w_ratio)

            # AttnRes-N
            attnres_n_ratio = norms[l][-2][ids>0].cpu()
            attnres_n_ratio_lis[l].append(attnres_n_ratio)

            # AttnResLn-N
            attnresln_n_ratio = norms[l][-1][ids>0].cpu()
            attnresln_n_ratio_lis[l].append(attnresln_n_ratio)

if not os.path.exists(f"./work/relation_with_freq/bert_{args.model}"):
    os.makedirs(f"./work/relation_with_freq/bert_{args.model}")

for l in range(layer_num):
    attn_w_ratio_tensor = torch.cat(attn_w_ratio_lis[l]).cpu()
    torch.save(attn_w_ratio_tensor, f"./work/relation_with_freq/bert_{args.model}/Attn_w_{l}layer_tensor")
    attn_n_ratio_tensor = torch.cat(attn_n_ratio_lis[l]).cpu()
    torch.save(attn_n_ratio_tensor, f"./work/relation_with_freq/bert_{args.model}/Attn_n_{l}layer_tensor")
    attnres_w_ratio_tensor = torch.cat(attnres_w_ratio_lis[l]).cpu()
    torch.save(attnres_w_ratio_tensor, f"./work/relation_with_freq/bert_{args.model}/AttnRes_w_{l}layer_tensor")
    attnres_n_ratio_tensor = torch.cat(attnres_n_ratio_lis[l]).cpu()
    torch.save(attnres_n_ratio_tensor, f"./work/relation_with_freq/bert_{args.model}/AttnRes_n_{l}layer_tensor")
    attnresln_n_ratio_tensor = torch.cat(attnresln_n_ratio_lis[l]).cpu()
    torch.save(attnresln_n_ratio_tensor, f"./work/relation_with_freq/bert_{args.model}/AttnResLn_n_{l}layer_tensor")