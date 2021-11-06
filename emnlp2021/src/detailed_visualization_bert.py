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

attn_w_all_ratio_lis =  [[] for _ in range(layer_num)]
attn_w_cls_ratio_lis =  [[] for _ in range(layer_num)]
attn_w_sep_ratio_lis =  [[] for _ in range(layer_num)]
attn_w_other_ratio_lis =  [[] for _ in range(layer_num)]
attn_w_mask_ratio_lis = [[] for _ in range(layer_num)]

attn_n_all_ratio_lis =  [[] for _ in range(layer_num)]
attn_n_cls_ratio_lis =  [[] for _ in range(layer_num)]
attn_n_sep_ratio_lis =  [[] for _ in range(layer_num)]
attn_n_other_ratio_lis =  [[] for _ in range(layer_num)]
attn_n_mask_ratio_lis = [[] for _ in range(layer_num)]

attnres_w_all_ratio_lis =  [[] for _ in range(layer_num)]
attnres_w_cls_ratio_lis =  [[] for _ in range(layer_num)]
attnres_w_sep_ratio_lis =  [[] for _ in range(layer_num)]
attnres_w_other_ratio_lis =  [[] for _ in range(layer_num)]
attnres_w_mask_ratio_lis = [[] for _ in range(layer_num)]

attnres_n_all_ratio_lis =  [[] for _ in range(layer_num)]
attnres_n_cls_ratio_lis =  [[] for _ in range(layer_num)]
attnres_n_sep_ratio_lis =  [[] for _ in range(layer_num)]
attnres_n_other_ratio_lis =  [[] for _ in range(layer_num)]
attnres_n_mask_ratio_lis = [[] for _ in range(layer_num)]

attnresln_n_all_ratio_lis =  [[] for _ in range(layer_num)]
attnresln_n_cls_ratio_lis =  [[] for _ in range(layer_num)]
attnresln_n_sep_ratio_lis =  [[] for _ in range(layer_num)]
attnresln_n_other_ratio_lis =  [[] for _ in range(layer_num)]
attnresln_n_mask_ratio_lis = [[] for _ in range(layer_num)]

with torch.no_grad():
    for i in tqdm(range(len(input_ids))):
        ids = torch.unsqueeze(input_ids[i], 0).to(device)
        
        # mask replacement
        word_type = ids.clone()
        word_type[word_type==cls_id] = -1
        word_type[word_type==sep_id] = -2
        word_type[word_type>0] = -3
        
        length = int(torch.sum(word_type==-3))
        mask_num = max(1, int(length * 0.12)) # CLS, SEP, PAD以外から12%(15%*80%) or 1つをマスク
        cand_lis = torch.nonzero((word_type==-3)[0]).squeeze().tolist()
        for mask_pos in random.sample(cand_lis, mask_num):
            ids[0, mask_pos] = mask_id
        
        mask = torch.unsqueeze(attention_mask[i], 0).to(device)
        type_ids = torch.unsqueeze(token_type_ids[i], 0).to(device)
        _, _, attentions, norms = model(ids, token_type_ids=type_ids, attention_mask=mask, output_attentions=True, output_norms=True)
        
        ids[ids==cls_id] = -1
        ids[ids==sep_id] = -2
        ids[ids==mask_id] = -5
        ids[ids>0] = -4
        
        for l in range(layer_num):
            head_num = len(attentions[l][0])
            avg_attn = torch.sum(attentions[l][0], dim=0) / head_num

            # Attn-W
            ratios = torch.sum(avg_attn, dim=1) - torch.diagonal(avg_attn)
            attn_w_all_ratio = float(torch.mean(ratios[ids[0]<0]))
            attn_w_all_ratio_lis[l].append(attn_w_all_ratio)
            attn_w_cls_ratio = float(torch.mean(ratios[ids[0]==-1]))
            attn_w_cls_ratio_lis[l].append(attn_w_cls_ratio)
            attn_w_sep_ratio = float(torch.mean(ratios[ids[0]==-2]))
            attn_w_sep_ratio_lis[l].append(attn_w_sep_ratio)
            if torch.sum(ratios[ids[0]==-4])>0:
                attn_w_other_ratio = float(torch.mean(ratios[ids[0]==-4]))
                attn_w_other_ratio_lis[l].append(attn_w_other_ratio)
            attn_w_mask_ratio = float(torch.mean(ratios[ids[0]==-5]))
            attn_w_mask_ratio_lis[l].append(attn_w_mask_ratio)

            # Attn-N
            attn_n_all_ratio = float(torch.mean(norms[l][-3][ids<0]))
            attn_n_all_ratio_lis[l].append(attn_n_all_ratio)
            attn_n_cls_ratio = float(torch.mean(norms[l][-3][ids==-1]))
            attn_n_cls_ratio_lis[l].append(attn_n_cls_ratio)
            attn_n_sep_ratio = float(torch.mean(norms[l][-3][ids==-2]))
            attn_n_sep_ratio_lis[l].append(attn_n_sep_ratio)
            if torch.sum(norms[l][-1][ids==-4])>0:
                attn_n_other_ratio = float(torch.mean(norms[l][-3][ids==-4]))
                attn_n_other_ratio_lis[l].append(attn_n_other_ratio)
            attn_n_mask_ratio = float(torch.mean(norms[l][-3][ids==-5]))
            attn_n_mask_ratio_lis[l].append(attn_n_mask_ratio)
            
            # AttnRes-W
            ratios = (torch.sum(avg_attn, dim=1) - torch.diagonal(avg_attn)) / 2
            attnres_w_all_ratio = float(torch.mean(ratios[ids[0]<0]))
            attnres_w_all_ratio_lis[l].append(attnres_w_all_ratio)
            attnres_w_cls_ratio = float(torch.mean(ratios[ids[0]==-1]))
            attnres_w_cls_ratio_lis[l].append(attnres_w_cls_ratio)
            attnres_w_sep_ratio = float(torch.mean(ratios[ids[0]==-2]))
            attnres_w_sep_ratio_lis[l].append(attnres_w_sep_ratio)
            if torch.sum(ratios[ids[0]==-4])>0:
                attnres_w_other_ratio = float(torch.mean(ratios[ids[0]==-4]))
                attnres_w_other_ratio_lis[l].append(attnres_w_other_ratio)
            attnres_w_mask_ratio = float(torch.mean(ratios[ids[0]==-5]))
            attnres_w_mask_ratio_lis[l].append(attnres_w_mask_ratio)

            # AttnRes-N
            attnres_n_all_ratio = float(torch.mean(norms[l][-2][ids<0]))
            attnres_n_all_ratio_lis[l].append(attnres_n_all_ratio)
            attnres_n_cls_ratio = float(torch.mean(norms[l][-2][ids==-1]))
            attnres_n_cls_ratio_lis[l].append(attnres_n_cls_ratio)
            attnres_n_sep_ratio = float(torch.mean(norms[l][-2][ids==-2]))
            attnres_n_sep_ratio_lis[l].append(attnres_n_sep_ratio)
            if torch.sum(norms[l][-1][ids==-4])>0:
                attnres_n_other_ratio = float(torch.mean(norms[l][-2][ids==-4]))
                attnres_n_other_ratio_lis[l].append(attnres_n_other_ratio)
            attnres_n_mask_ratio = float(torch.mean(norms[l][-2][ids==-5]))
            attnres_n_mask_ratio_lis[l].append(attnres_n_mask_ratio)

            # AttnResLn-N
            attnresln_n_all_ratio = float(torch.mean(norms[l][-1][ids<0]))
            attnresln_n_all_ratio_lis[l].append(attnresln_n_all_ratio)
            attnresln_n_cls_ratio = float(torch.mean(norms[l][-1][ids==-1]))
            attnresln_n_cls_ratio_lis[l].append(attnresln_n_cls_ratio)
            attnresln_n_sep_ratio = float(torch.mean(norms[l][-1][ids==-2]))
            attnresln_n_sep_ratio_lis[l].append(attnresln_n_sep_ratio)
            if torch.sum(norms[l][-1][ids==-4])>0:
                attnresln_n_other_ratio = float(torch.mean(norms[l][-1][ids==-4]))
                attnresln_n_other_ratio_lis[l].append(attnresln_n_other_ratio)
            attnresln_n_mask_ratio = float(torch.mean(norms[l][-1][ids==-5]))
            attnresln_n_mask_ratio_lis[l].append(attnresln_n_mask_ratio)

        del norms, _
        torch.cuda.empty_cache()

def convert_layer_avg(ratio_lis):
    lis = []
    for l in range(layer_num):        
        lis.append(float(sum(ratio_lis[l])/len(ratio_lis[l])))
    return lis

if not os.path.exists(f"./work/bert_{args.model}/avg"):
    os.makedirs(f"./work/bert_{args.model}/avg")

# Attn-W 
avg_attn_w_all_ratio_lis = convert_layer_avg(attn_w_all_ratio_lis)
torch.save(avg_attn_w_all_ratio_lis, f"./work/bert_{args.model}/avg/Attn_w_all")
avg_attn_w_cls_ratio_lis = convert_layer_avg(attn_w_cls_ratio_lis)
torch.save(avg_attn_w_cls_ratio_lis, f"./work/bert_{args.model}/avg/Attn_w_cls")
avg_attn_w_sep_ratio_lis = convert_layer_avg(attn_w_sep_ratio_lis)
torch.save(avg_attn_w_sep_ratio_lis, f"./work/bert_{args.model}/avg/Attn_w_sep")
avg_attn_w_mask_ratio_lis = convert_layer_avg(attn_w_mask_ratio_lis)
torch.save(avg_attn_w_mask_ratio_lis, f"./work/bert_{args.model}/avg/Attn_w_mask")
avg_attn_w_other_ratio_lis = convert_layer_avg(attn_w_other_ratio_lis)
torch.save(avg_attn_w_other_ratio_lis, f"./work/bert_{args.model}/avg/Attn_w_other")

# Attn-N
avg_attn_n_all_ratio_lis = convert_layer_avg(attn_n_all_ratio_lis)
torch.save(avg_attn_n_all_ratio_lis, f"./work/bert_{args.model}/avg/Attn_n_all")
avg_attn_n_cls_ratio_lis = convert_layer_avg(attn_n_cls_ratio_lis)
torch.save(avg_attn_n_cls_ratio_lis, f"./work/bert_{args.model}/avg/Attn_n_cls")
avg_attn_n_sep_ratio_lis = convert_layer_avg(attn_n_sep_ratio_lis)
torch.save(avg_attn_n_sep_ratio_lis, f"./work/bert_{args.model}/avg/Attn_n_sep")
avg_attn_n_mask_ratio_lis = convert_layer_avg(attn_n_mask_ratio_lis)
torch.save(avg_attn_n_mask_ratio_lis, f"./work/bert_{args.model}/avg/Attn_n_mask")
avg_attn_n_other_ratio_lis = convert_layer_avg(attn_n_other_ratio_lis)
torch.save(avg_attn_n_other_ratio_lis, f"./work/bert_{args.model}/avg/Attn_n_other")

# AttnRes-W
avg_attnres_w_all_ratio_lis = convert_layer_avg(attnres_w_all_ratio_lis)
torch.save(avg_attnres_w_all_ratio_lis, f"./work/bert_{args.model}/avg/AttnRes_w_all")
avg_attnres_w_cls_ratio_lis = convert_layer_avg(attnres_w_cls_ratio_lis)
torch.save(avg_attnres_w_cls_ratio_lis, f"./work/bert_{args.model}/avg/AttnRes_w_cls")
avg_attnres_w_sep_ratio_lis = convert_layer_avg(attnres_w_sep_ratio_lis)
torch.save(avg_attnres_w_sep_ratio_lis, f"./work/bert_{args.model}/avg/AttnRes_w_sep")
avg_attnres_w_mask_ratio_lis = convert_layer_avg(attnres_w_mask_ratio_lis)
torch.save(avg_attnres_w_mask_ratio_lis, f"./work/bert_{args.model}/avg/AttnRes_w_mask")
avg_attnres_w_other_ratio_lis = convert_layer_avg(attnres_w_other_ratio_lis)
torch.save(avg_attnres_w_other_ratio_lis, f"./work/bert_{args.model}/avg/AttnRes_w_other")

# AttnRes-N
avg_attnres_n_all_ratio_lis = convert_layer_avg(attnres_n_all_ratio_lis)
torch.save(avg_attnres_n_all_ratio_lis, f"./work/bert_{args.model}/avg/AttnRes_n_all")
avg_attnres_n_cls_ratio_lis = convert_layer_avg(attnres_n_cls_ratio_lis)
torch.save(avg_attnres_n_cls_ratio_lis, f"./work/bert_{args.model}/avg/AttnRes_n_cls")
avg_attnres_n_sep_ratio_lis = convert_layer_avg(attnres_n_sep_ratio_lis)
torch.save(avg_attnres_n_sep_ratio_lis, f"./work/bert_{args.model}/avg/AttnRes_n_sep")
avg_attnres_n_mask_ratio_lis = convert_layer_avg(attnres_n_mask_ratio_lis)
torch.save(avg_attnres_n_mask_ratio_lis, f"./work/bert_{args.model}/avg/AttnRes_n_mask")
avg_attnres_n_other_ratio_lis = convert_layer_avg(attnres_n_other_ratio_lis)
torch.save(avg_attnres_n_other_ratio_lis, f"./work/bert_{args.model}/avg/AttnRes_n_other")

# AttnResLn-N
avg_attnresln_n_all_ratio_lis = convert_layer_avg(attnresln_n_all_ratio_lis)
torch.save(avg_attnresln_n_all_ratio_lis, f"./work/bert_{args.model}/avg/AttnResLn_n_all")
avg_attnresln_n_cls_ratio_lis = convert_layer_avg(attnresln_n_cls_ratio_lis)
torch.save(avg_attnresln_n_cls_ratio_lis, f"./work/bert_{args.model}/avg/AttnResLn_n_cls")
avg_attnresln_n_sep_ratio_lis = convert_layer_avg(attnresln_n_sep_ratio_lis)
torch.save(avg_attnresln_n_sep_ratio_lis, f"./work/bert_{args.model}/avg/AttnResLn_n_sep")
avg_attnresln_n_mask_ratio_lis = convert_layer_avg(attnresln_n_mask_ratio_lis)
torch.save(avg_attnresln_n_mask_ratio_lis, f"./work/bert_{args.model}/avg/AttnResLn_n_mask")
avg_attnresln_n_other_ratio_lis = convert_layer_avg(attnresln_n_other_ratio_lis)
torch.save(avg_attnresln_n_other_ratio_lis, f"./work/bert_{args.model}/avg/AttnResLn_n_other")