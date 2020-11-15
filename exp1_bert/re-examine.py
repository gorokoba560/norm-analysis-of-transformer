import pickle
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel


device = "cuda" if torch.cuda.is_available() else "cpu"

input_ids = torch.load("./work/input_ids.pt")
attention_mask = torch.load('./work/attention_mask.pt')
token_type_ids = torch.load('./work/token_type_ids.pt')
seq_len = torch.load('./work/seq_len.pt')

model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

cls_id = torch.tensor(tokenizer.convert_tokens_to_ids("[CLS]"))
sep_id = torch.tensor(tokenizer.convert_tokens_to_ids("[SEP]"))
comma_id = torch.tensor(tokenizer.convert_tokens_to_ids(","))
period_id = torch.tensor(tokenizer.convert_tokens_to_ids("."))

cls_attention_lis =  [[[] for _ in range(12)] for _ in range(12)]
cls_fx_lis =  [[[] for _ in range(12)] for _ in range(12)]
cls_afx_lis =  [[[] for _ in range(12)] for _ in range(12)]
sep_attention_lis =  [[[] for _ in range(12)] for _ in range(12)]
sep_fx_lis =  [[[] for _ in range(12)] for _ in range(12)]
sep_afx_lis =  [[[] for _ in range(12)] for _ in range(12)]
cp_attention_lis =  [[[] for _ in range(12)] for _ in range(12)]
cp_fx_lis =  [[[] for _ in range(12)] for _ in range(12)]
cp_afx_lis =  [[[] for _ in range(12)] for _ in range(12)]
other_attention_lis =  [[[] for _ in range(12)] for _ in range(12)]
other_fx_lis =  [[[] for _ in range(12)] for _ in range(12)]
other_afx_lis =  [[[] for _ in range(12)] for _ in range(12)]

with torch.no_grad():
    for i in tqdm(range(len(input_ids))):
        ids = torch.unsqueeze(input_ids[i], 0).to(device)
        mask = torch.unsqueeze(attention_mask[i], 0).to(device)
        type_ids = torch.unsqueeze(token_type_ids[i], 0).to(device)
        _, _, attentions, norms = model(ids, token_type_ids=type_ids, attention_mask=mask, output_attentions=True, output_norms=True)
        
        ids[ids==cls_id] = -1
        ids[ids==sep_id] = -2
        ids[ids==comma_id] = -3
        ids[ids==period_id] = -3
        ids[ids>0] = -4
        
        for l in range(12):
            for h in range(12):
                cls_attn = float(torch.mean(attentions[l][0,h,:seq_len[i],ids[0]==-1].sum(1)).cpu())
                cls_attention_lis[l][h].append(cls_attn)
                cls_fx = float(torch.mean(norms[l][0][0,h,ids[0]==-1].mean(-1)).cpu())
                cls_fx_lis[l][h].append(cls_fx)
                cls_afx = float(torch.mean(norms[l][1][0,h,0:seq_len[i],ids[0]==-1].sum(1)).cpu())
                cls_afx_lis[l][h].append(cls_afx)
                
                sep_attn = float(torch.mean(attentions[l][0,h,:seq_len[i],ids[0]==-2].sum(1)).cpu())
                sep_attention_lis[l][h].append(sep_attn)
                sep_fx = float(torch.mean(norms[l][0][0,h,ids[0]==-2].mean(-1)).cpu())
                sep_fx_lis[l][h].append(sep_fx)
                sep_afx = float(torch.mean(norms[l][1][0,h,0:seq_len[i],ids[0]==-2].sum(1)).cpu())
                sep_afx_lis[l][h].append(sep_afx)

                cp_attn = float(torch.mean(attentions[l][0,h,:seq_len[i],ids[0]==-3].sum(1)).cpu())
                cp_attention_lis[l][h].append(cp_attn)
                cp_fx = float(torch.mean(norms[l][0][0,h,ids[0]==-3].mean(-1)).cpu())
                cp_fx_lis[l][h].append(cp_fx)
                cp_afx = float(torch.mean(norms[l][1][0,h,0:seq_len[i],ids[0]==-3].sum(1)).cpu())
                cp_afx_lis[l][h].append(cp_afx)

                other_attn = float(torch.mean(attentions[l][0,h,:seq_len[i],ids[0]==-4].sum(1)).cpu())
                other_attention_lis[l][h].append(other_attn)
                other_fx = float(torch.mean(norms[l][0][0,h,ids[0]==-4].mean(-1)).cpu())
                other_fx_lis[l][h].append(other_fx)
                other_afx = float(torch.mean(norms[l][1][0,h,0:seq_len[i],ids[0]==-4].sum(1)).cpu())
                other_afx_lis[l][h].append(other_afx)

        del attentions, norms, _
        torch.cuda.empty_cache()

torch.save(cls_attention_lis, "./work/re-examine/cls_a_instance_lis")
torch.save(cls_fx_lis, "./work/re-examine/cls_fx_instance_lis")
torch.save(cls_afx_lis, "./work/re-examine/cls_afx_instance_lis")
torch.save(sep_attention_lis, "./work/re-examine/sep_a_instance_lis")
torch.save(sep_fx_lis, "./work/re-examine/sep_fx_instance_lis")
torch.save(sep_afx_lis, "./work/re-examine/sep_afx_instance_lis")
torch.save(cp_attention_lis, "./work/re-examine/cp_a_instance_lis")
torch.save(cp_fx_lis, "./work/re-examine/cp_fx_instance_lis")
torch.save(cp_afx_lis, "./work/re-examine/cp_afx_instance_lis")
torch.save(other_attention_lis, "./work/re-examine/other_a_instance_lis")
torch.save(other_fx_lis, "./work/re-examine/other_fx_instance_lis")
torch.save(other_afx_lis, "./work/re-examine/other_afx_instance_lis")


def convert_head_avg(a_lis, fx_lis, afx_lis):
    a = [[] for _ in range(12)]
    fx = [[] for _ in range(12)]
    afx = [[] for _ in range(12)]
    for l in range(12):
        for h in range(12):
            a[l].append(float(sum(a_lis[l][h])/len(a_lis[l][h])))
            fx[l].append(float(sum(fx_lis[l][h])/len(fx_lis[l][h])))
            afx[l].append(float(sum(afx_lis[l][h])/len(afx_lis[l][h])))
    return a, fx, afx

cls_a_head_lis, cls_fx_head_lis, cls_afx_head_lis = convert_head_avg(cls_attention_lis, cls_fx_lis, cls_afx_lis)
sep_a_head_lis, sep_fx_head_lis, sep_afx_head_lis = convert_head_avg(sep_attention_lis, sep_fx_lis, sep_afx_lis)
cp_a_head_lis, cp_fx_head_lis, cp_afx_head_lis = convert_head_avg(cp_attention_lis, cp_fx_lis, cp_afx_lis)
other_a_head_lis, other_fx_head_lis, other_afx_head_lis = convert_head_avg(other_attention_lis, other_fx_lis, other_afx_lis)
torch.save(cls_a_head_lis, "./work/re-examine/cls_a_head_lis")
torch.save(cls_fx_head_lis, "./work/re-examine/cls_fx_head_lis")
torch.save(cls_afx_head_lis, "./work/re-examine/cls_afx_head_lis")
torch.save(sep_a_head_lis, "./work/re-examine/sep_a_head_lis")
torch.save(sep_fx_head_lis, "./work/re-examine/sep_fx_head_lis")
torch.save(sep_afx_head_lis, "./work/re-examine/sep_afx_head_lis")
torch.save(cp_a_head_lis, "./work/re-examine/cp_a_head_lis")
torch.save(cp_fx_head_lis, "./work/re-examine/cp_fx_head_lis")
torch.save(cp_afx_head_lis, "./work/re-examine/cp_afx_head_lis")
torch.save(other_a_head_lis, "./work/re-examine/other_a_head_lis")
torch.save(other_fx_head_lis, "./work/re-examine/other_fx_head_lis")
torch.save(other_afx_head_lis, "./work/re-examine/other_afx_head_lis")


def convert_layer_avg(a_lis, fx_lis, afx_lis):
    a_head_avg, fx_head_avg, afx_head_avg = convert_head_avg(a_lis, fx_lis, afx_lis)
    a = []
    fx = []
    afx = []
    for l in range(12):
        a.append(float(sum(a_head_avg[l])/len(a_head_avg[l])))
        fx.append(float(sum(fx_head_avg[l])/len(fx_head_avg[l])))
        afx.append(float(sum(afx_head_avg[l])/len(afx_head_avg[l])))
    return a, fx, afx

cls_a_layer_lis, cls_fx_layer_lis, cls_afx_layer_lis = convert_layer_avg(cls_attention_lis, cls_fx_lis, cls_afx_lis)
sep_a_layer_lis, sep_fx_layer_lis, sep_afx_layer_lis = convert_layer_avg(sep_attention_lis, sep_fx_lis, sep_afx_lis)
cp_a_layer_lis, cp_fx_layer_lis, cp_afx_layer_lis = convert_layer_avg(cp_attention_lis, cp_fx_lis, cp_afx_lis)
other_a_layer_lis, other_fx_layer_lis, other_afx_layer_lis = convert_layer_avg(other_attention_lis, other_fx_lis, other_afx_lis)
torch.save(cls_a_layer_lis, "./work/re-examine/cls_a_layer_lis")
torch.save(cls_fx_layer_lis, "./work/re-examine/cls_fx_layer_lis")
torch.save(cls_afx_layer_lis, "./work/re-examine/cls_afx_layer_lis")
torch.save(sep_a_layer_lis, "./work/re-examine/sep_a_layer_lis")
torch.save(sep_fx_layer_lis, "./work/re-examine/sep_fx_layer_lis")
torch.save(sep_afx_layer_lis, "./work/re-examine/sep_afx_layer_lis")
torch.save(cp_a_layer_lis, "./work/re-examine/cp_a_layer_lis")
torch.save(cp_fx_layer_lis, "./work/re-examine/cp_fx_layer_lis")
torch.save(cp_afx_layer_lis, "./work/re-examine/cp_afx_layer_lis")
torch.save(other_a_layer_lis, "./work/re-examine/other_a_layer_lis")
torch.save(other_fx_layer_lis, "./work/re-examine/other_fx_layer_lis")
torch.save(other_afx_layer_lis, "./work/re-examine/other_afx_layer_lis")