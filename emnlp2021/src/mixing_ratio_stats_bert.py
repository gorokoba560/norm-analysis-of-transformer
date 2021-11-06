import pickle
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
import random
import argparse
import os


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

input_ids = torch.load("./work/input_ids.pt")
attention_mask = torch.load('./work/attention_mask.pt')
token_type_ids = torch.load('./work/token_type_ids.pt')
seq_len = torch.load('./work/seq_len.pt')

model = BertModel.from_pretrained(pretrained).to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained(pretrained)

cls_id = torch.tensor(tokenizer.convert_tokens_to_ids("[CLS]"))
sep_id = torch.tensor(tokenizer.convert_tokens_to_ids("[SEP]"))
mask_id = torch.tensor(tokenizer.convert_tokens_to_ids("[MASK]"))

all_attn_w_ratio_lis =  []
all_attn_n_ratio_lis =  []
all_attnres_w_ratio_lis =  []
all_attnres_n_ratio_lis =  []
all_attnresln_n_ratio_lis =  []

with torch.no_grad():
    for i in tqdm(range(len(input_ids))):
        ids = torch.unsqueeze(input_ids[i], 0).to(device)
        
        # mask replacement
        word_type = ids.clone()
        word_type[word_type==cls_id] = -1
        word_type[word_type==sep_id] = -2
        word_type[word_type>0] = -3
        
        length = int(torch.sum(word_type==-3))
        mask_num = max(1, int(length * 0.12)) # mask 12% of tokens excluding CLS, SEP, and PAD
        cand_lis = torch.nonzero((word_type==-3)[0]).squeeze().tolist()
        for mask_pos in random.sample(cand_lis, mask_num):
            ids[0, mask_pos] = mask_id
        
        mask = torch.unsqueeze(attention_mask[i], 0).to(device)
        type_ids = torch.unsqueeze(token_type_ids[i], 0).to(device)
        _, _, attentions, norms = model(ids, 
                                        token_type_ids=type_ids, 
                                        attention_mask=mask, 
                                        output_attentions=True, 
                                        output_norms=True
                                        )
        
        for l in range(layer_num):
            # Attn-W
            head_num = len(attentions[l][0])
            avg_attn = torch.sum(attentions[l][0], dim=0)/head_num
            ratios = torch.sum(avg_attn, dim=1) - torch.diagonal(avg_attn)
            attn_w_ratio = ratios[ids[0]!=0].cpu().tolist()
            all_attn_w_ratio_lis.append(attn_w_ratio)

            # Attn-N
            attn_n_ratio = norms[l][-3][ids!=0].cpu()
            all_attn_n_ratio_lis.append(attn_n_ratio)

            # AttnRes-W
            ratios = (torch.sum(avg_attn, dim=1) - torch.diagonal(avg_attn)) / 2
            attnres_w_ratio = ratios[ids[0]!=0].cpu().tolist()
            all_attnres_w_ratio_lis.append(attnres_w_ratio)

            # AttnRes-N
            attnres_n_ratio = norms[l][-2][ids!=0].cpu()
            all_attnres_n_ratio_lis.append(attnres_n_ratio)

            # AttnResLn-N
            attnresln_n_ratio = norms[l][-1][ids!=0].cpu()
            all_attnresln_n_ratio_lis.append(attnresln_n_ratio)

if not os.path.exists(f"./work/bert_{args.model}"):
    os.makedirs(f"./work/bert_{args.model}")

torch.save(all_attn_w_ratio_lis, f"./work/bert_{args.model}/Attn_w_mixing_ratio")
torch.save(all_attn_n_ratio_lis, f"./work/bert_{args.model}/Attn_n_mixing_ratio")
torch.save(all_attnres_w_ratio_lis, f"./work/bert_{args.model}/AttnRes_w_mixing_ratio")
torch.save(all_attnres_n_ratio_lis, f"./work/bert_{args.model}/AttnRes_n_mixing_ratio")
torch.save(all_attnresln_n_ratio_lis, f"./work/bert_{args.model}/AttnResLn_n_mixing_ratio")