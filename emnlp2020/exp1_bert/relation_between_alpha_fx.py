import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"

input_ids = torch.load("./work/input_ids.pt")
attention_mask = torch.load('./work/attention_mask.pt')
token_type_ids = torch.load('./work/token_type_ids.pt')
seq_len = torch.load('./work/seq_len.pt')

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
cls_id = torch.tensor(tokenizer.convert_tokens_to_ids("[CLS]"))
sep_id = torch.tensor(tokenizer.convert_tokens_to_ids("[SEP]"))
comma_id = torch.tensor(tokenizer.convert_tokens_to_ids(","))
period_id = torch.tensor(tokenizer.convert_tokens_to_ids("."))

model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

cls_attention_lis =  []
cls_fx_lis =  []

sep_attention_lis =  []
sep_fx_lis = []

cp_attention_lis =  []
cp_fx_lis =  []

other_attention_lis =  []
other_fx_lis =  []

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
                cls_attn = attentions[l][0,h,:seq_len[i],ids[0]==-1].view(-1).cpu()
                cls_attention_lis.append(cls_attn)
                cls_fx = norms[l][0][0,h,ids[0]==-1].repeat(seq_len[i]).cpu()
                cls_fx_lis.append(cls_fx)
                
                sep_attn = attentions[l][0,h,:seq_len[i],ids[0]==-2].view(-1).cpu()
                sep_attention_lis.append(sep_attn)
                sep_fx = norms[l][0][0,h,ids[0]==-2].repeat(seq_len[i]).cpu()
                sep_fx_lis.append(sep_fx)
                
                cp_attn = attentions[l][0,h,:seq_len[i],ids[0]==-3].view(-1).cpu()
                cp_attention_lis.append(cp_attn)
                cp_fx = norms[l][0][0,h,ids[0]==-3].repeat(seq_len[i]).cpu()
                cp_fx_lis.append(cp_fx)
                
                other_attn = attentions[l][0,h,:seq_len[i],ids[0]==-4].view(-1).cpu()
                other_attention_lis.append(other_attn)
                other_fx = norms[l][0][0,h,ids[0]==-4].repeat(seq_len[i]).cpu()
                other_fx_lis.append(other_fx)

        del attentions, norms, _
        torch.cuda.empty_cache()

cls_attention_lis = torch.cat(cls_attention_lis)
torch.save(cls_attention_lis, "./work/relation_between_alpha_fx/cls_attention.pt")
del cls_attention_lis

cls_fx_lis = torch.cat(cls_fx_lis)
torch.save(cls_fx_lis, "./work/relation_between_alpha_fx/cls_fx.pt")
del cls_fx_lis

sep_attention_lis = torch.cat(sep_attention_lis)
torch.save(sep_attention_lis, "./work/relation_between_alpha_fx/sep_attention.pt")
del sep_attention_lis

sep_fx_lis = torch.cat(sep_fx_lis)
torch.save(sep_fx_lis, "./work/relation_between_alpha_fx/sep_fx.pt")
del sep_fx_lis

cp_attention_lis = torch.cat(cp_attention_lis)
torch.save(cp_attention_lis, "./work/relation_between_alpha_fx/cp_attention.pt")
del cp_attention_lis

cp_fx_lis = torch.cat(cp_fx_lis)
torch.save(cp_fx_lis, "./work/relation_between_alpha_fx/cp_fx.pt")
del cp_fx_lis

other_attention_lis = torch.cat(other_attention_lis)
torch.save(other_attention_lis, "./work/relation_between_alpha_fx/other_attention.pt")
del other_attention_lis

other_fx_lis = torch.cat(other_fx_lis)
torch.save(other_fx_lis, "./work/relation_between_alpha_fx/other_fx.pt")
del other_fx_lis