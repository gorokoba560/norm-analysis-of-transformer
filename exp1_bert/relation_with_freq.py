import torch
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import BertTokenizer, BertModel


device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
pad_id = torch.tensor(tokenizer.convert_tokens_to_ids("[PAD]"))

seq_len = torch.load('./work/seq_len.pt')
input_ids = torch.load("./work/input_ids.pt")
attention_mask = torch.load('./work/attention_mask.pt')
token_type_ids = torch.load('./work/token_type_ids.pt')
seq_len = torch.load('./work/seq_len.pt')

input_rank = input_ids.clone()
input_count = input_ids.clone()
input_memo = input_ids.clone()

with open ("./data/rank_count_ids.txt") as f:
    for line in f:
        rank, count, ids = map(int, line.split())
        input_rank[input_ids==ids] = rank
        input_count[input_ids==ids] = count
        input_memo[input_ids==ids] = -1

input_rank[input_memo==0] = -1
input_count[input_memo==0] = -1

input_rank[input_memo>0] = rank + 1
input_count[input_memo>0] = 0

rank = input_rank[input_ids!=pad_id]
count = input_count[input_ids!=pad_id]

torch.save(input_rank, './work/relation_with_freq/input_rank.pt')
torch.save(input_count, './work/relation_with_freq/input_count.pt')

model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

a_lis = []
fx_lis = []

with torch.no_grad():
    for i in tqdm(range(len(input_ids))):
        ids = torch.unsqueeze(input_ids[i], 0).to(device)
        mask = torch.unsqueeze(attention_mask[i], 0).to(device)
        type_ids = torch.unsqueeze(token_type_ids[i], 0).to(device)
        _, _, attentions, norms = model(ids, token_type_ids=type_ids, attention_mask=mask, output_attentions=True, output_norms=True)
        
        for l in range(12):
            a = torch.mean(torch.mean(attentions[l][0,:,0:seq_len[i],0:seq_len[i]], dim=0), dim=0)
            fx = torch.mean(norms[l][0][0,:,0:seq_len[i]], dim=0)

            if l==0:
                memo_a = a.clone()
                memo_fx = fx.clone()
            else:
                memo_a += a
                memo_fx += fx

        a_lis.append(memo_a/12)
        fx_lis.append(memo_fx/12)
        
        del attentions, norms, _
        torch.cuda.empty_cache()

a_tensor = torch.cat(a_lis).cpu()
fx_tensor = torch.cat(fx_lis).cpu()

assert len(a_tensor) == len(fx_tensor)
assert len(fx_tensor) == torch.sum(input_ids!=pad_id)
torch.save(a_tensor, "./work/relation_with_freq/all_token_mean_a.pt")
torch.save(fx_tensor, "./work/relation_with_freq/all_token_mean_fx.pt")