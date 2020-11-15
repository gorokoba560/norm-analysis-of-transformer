from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel


device = "cuda" if torch.cuda.is_available() else "cpu"

seq_len = torch.load('./work/seq_len.pt')
input_ids = torch.load('./work/input_ids.pt')
attention_mask = torch.load('./work/attention_mask.pt')
token_type_ids = torch.load('./work/token_type_ids.pt')

model = BertModel.from_pretrained("bert-base-uncased").to(device)
model.eval()

x_norm_lis = [[] for _ in range(12)]
fx_norm_lis = [[[] for _ in range(12)] for _ in range(12)]

with torch.no_grad():
    for i in tqdm(range(len(input_ids))):
        ids = torch.unsqueeze(input_ids[i], 0).to(device)
        mask = torch.unsqueeze(attention_mask[i], 0).to(device)
        type_ids = torch.unsqueeze(token_type_ids[i], 0).to(device)
        _, _, hidden_states, norms = model(ids, token_type_ids=type_ids, attention_mask=mask, output_hidden_states=True, output_norms=True)
        for l in range(12):
            for h in range(12):
                fx_norm = norms[l][0][0,h,:seq_len[i]].cpu()
                fx_norm_lis[l][h] += fx_norm.tolist()
            x_norm = torch.norm(hidden_states[l][0,:seq_len[i]].cpu(), dim=-1)
            x_norm_lis[l] += x_norm.tolist()
        del hidden_states, norms
        torch.cuda.empty_cache()

torch.save(x_norm_lis, "./work/check_dispersion/x_norm_lis.pt")
torch.save(fx_norm_lis, "./work/check_dispersion/fx_norm_lis.pt")