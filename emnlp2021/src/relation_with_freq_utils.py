import os
import torch
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def layer_num_from_bert_size(size):
    if size == "base":
        layer_num = 12
    elif size == "large":
        layer_num = 24
    elif size == "medium":
        layer_num = 8
    elif size == "small" or size == "mini":
        layer_num = 4
    elif size == "tiny":
        layer_num = 2
    else:
        print("args 'size' should be 'base', 'large', 'medium', 'small', 'mini', or 'tiny'.")
        return None
    return layer_num


def rank_spearman(size):
    layer_num = layer_num_from_bert_size(size)
    if layer_num == None:
        return

    rank = torch.load(f'./work/relation_with_freq/rank.pt')

    for l in range(layer_num):
        attn_w_ratio_tensor = torch.load(f"./work/relation_with_freq/bert_{size}/Attn_w_{l}layer_tensor")
        attn_n_ratio_tensor = torch.load(f"./work/relation_with_freq/bert_{size}/Attn_n_{l}layer_tensor")
        attnres_w_ratio_tensor = torch.load(f"./work/relation_with_freq/bert_{size}/AttnRes_w_{l}layer_tensor")
        attnres_n_ratio_tensor = torch.load(f"./work/relation_with_freq/bert_{size}/AttnRes_n_{l}layer_tensor")
        attnresln_n_ratio_tensor = torch.load(f"./work/relation_with_freq/bert_{size}/AttnResLn_n_{l}layer_tensor")

        if l==0:
            avg_attn_w_ratio_tensor = attn_w_ratio_tensor.clone()
            avg_attn_n_ratio_tensor = attn_n_ratio_tensor.clone()
            avg_attnres_w_ratio_tensor = attnres_w_ratio_tensor.clone()
            avg_attnres_n_ratio_tensor = attnres_n_ratio_tensor.clone()
            avg_attnresln_n_ratio_tensor = attnresln_n_ratio_tensor.clone()
        else:
            avg_attn_w_ratio_tensor += attn_w_ratio_tensor.clone()
            avg_attn_n_ratio_tensor += attn_n_ratio_tensor.clone()
            avg_attnres_w_ratio_tensor += attnres_w_ratio_tensor.clone()
            avg_attnres_n_ratio_tensor += attnres_n_ratio_tensor.clone()
            avg_attnresln_n_ratio_tensor += attnresln_n_ratio_tensor.clone()
    avg_attn_w_ratio_tensor /= layer_num
    avg_attn_n_ratio_tensor /= layer_num
    avg_attnres_w_ratio_tensor /= layer_num
    avg_attnres_n_ratio_tensor /= layer_num
    avg_attnresln_n_ratio_tensor /= layer_num
        
    results = spearmanr(rank, avg_attn_w_ratio_tensor)
    print(f"Attn-W:\t\t{results.correlation:.3f}")
        
    results = spearmanr(rank, avg_attn_n_ratio_tensor)
    print(f"Attn-N:\t\t{results.correlation:.3f}")
    
    results = spearmanr(rank, avg_attnres_w_ratio_tensor)
    print(f"AttnRes-W:\t{results.correlation:.3f}")
    
    results = spearmanr(rank, avg_attnres_n_ratio_tensor)
    print(f"AttnRes-N:\t{results.correlation:.3f}")
    
    results = spearmanr(rank, avg_attnresln_n_ratio_tensor)
    print(f"AttnResLn-N:\t{results.correlation:.3f}")


def rank_spearman_without_specials(size):
    layer_num = layer_num_from_bert_size(size)
    if layer_num == None:
        return

    rank = torch.load(f'./work/relation_with_freq/rank.pt')
    input_ids = torch.load("./work/input_ids.pt")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    cls_id = torch.tensor(tokenizer.convert_tokens_to_ids("[CLS]"))
    sep_id = torch.tensor(tokenizer.convert_tokens_to_ids("[SEP]"))

    for l in range(layer_num):
        attn_w_ratio_tensor = torch.load(f"./work/relation_with_freq/bert_{size}/Attn_w_{l}layer_tensor")
        attn_n_ratio_tensor = torch.load(f"./work/relation_with_freq/bert_{size}/Attn_n_{l}layer_tensor")
        attnres_w_ratio_tensor = torch.load(f"./work/relation_with_freq/bert_{size}/AttnRes_w_{l}layer_tensor")
        attnres_n_ratio_tensor = torch.load(f"./work/relation_with_freq/bert_{size}/AttnRes_n_{l}layer_tensor")
        attnresln_n_ratio_tensor = torch.load(f"./work/relation_with_freq/bert_{size}/AttnResLn_n_{l}layer_tensor")

        if l==0:
            avg_attn_w_ratio_tensor = attn_w_ratio_tensor.clone()
            avg_attn_n_ratio_tensor = attn_n_ratio_tensor.clone()
            avg_attnres_w_ratio_tensor = attnres_w_ratio_tensor.clone()
            avg_attnres_n_ratio_tensor = attnres_n_ratio_tensor.clone()
            avg_attnresln_n_ratio_tensor = attnresln_n_ratio_tensor.clone()
        else:
            avg_attn_w_ratio_tensor += attn_w_ratio_tensor.clone()
            avg_attn_n_ratio_tensor += attn_n_ratio_tensor.clone()
            avg_attnres_w_ratio_tensor += attnres_w_ratio_tensor.clone()
            avg_attnres_n_ratio_tensor += attnres_n_ratio_tensor.clone()
            avg_attnresln_n_ratio_tensor += attnresln_n_ratio_tensor.clone()
    avg_attn_w_ratio_tensor /= layer_num
    avg_attn_n_ratio_tensor /= layer_num
    avg_attnres_w_ratio_tensor /= layer_num
    avg_attnres_n_ratio_tensor /= layer_num
    avg_attnresln_n_ratio_tensor /= layer_num
        
    ids = input_ids[input_ids!=0]

    # remove ratios corresponding to special tokens
    for i in [cls_id, sep_id]:
        avg_attn_w_ratio_tensor = avg_attn_w_ratio_tensor[ids != i]
        avg_attn_n_ratio_tensor = avg_attn_n_ratio_tensor[ids != i]
        avg_attnres_w_ratio_tensor = avg_attnres_w_ratio_tensor[ids != i]
        avg_attnres_n_ratio_tensor = avg_attnres_n_ratio_tensor[ids != i]
        avg_attnresln_n_ratio_tensor = avg_attnresln_n_ratio_tensor[ids != i]
        rank = rank[ids != i]
        ids = ids[ids != i]

    results = spearmanr(rank, avg_attn_w_ratio_tensor)
    print(f"Attn-W:\t\t{results.correlation:.3f}")
        
    results = spearmanr(rank, avg_attn_n_ratio_tensor)
    print(f"Attn-N:\t\t{results.correlation:.3f}")
    
    results = spearmanr(rank, avg_attnres_w_ratio_tensor)
    print(f"AttnRes-W:\t{results.correlation:.3f}")
    
    results = spearmanr(rank, avg_attnres_n_ratio_tensor)
    print(f"AttnRes-N:\t{results.correlation:.3f}")
    
    results = spearmanr(rank, avg_attnresln_n_ratio_tensor)
    print(f"AttnResLn-N:\t{results.correlation:.3f}")


def viz_rank(size):
    layer_num = layer_num_from_bert_size(size)
    if layer_num == None:
        return
    
    # set the font size
    fs = 34
    fs2 = 25

    # set the color map & style
    cmap = plt.get_cmap("tab10")
    plt.style.use('seaborn-darkgrid')

    input_ids = torch.load("./work/input_ids.pt")
    ids = input_ids[input_ids!=0]
    rank = torch.load(f'./work/relation_with_freq/rank.pt')

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    cls_id = torch.tensor(tokenizer.convert_tokens_to_ids("[CLS]"))
    sep_id = torch.tensor(tokenizer.convert_tokens_to_ids("[SEP]"))
    comma_id = torch.tensor(tokenizer.convert_tokens_to_ids(","))
    period_id = torch.tensor(tokenizer.convert_tokens_to_ids("."))

    for l in range(layer_num):     
        ratio_tensor = torch.load(f"./work/relation_with_freq/bert_{size}/AttnResLn_n_{l}layer_tensor")   
        if l==0:
            avg_ratio_tensor = ratio_tensor.clone()
        else:
            avg_ratio_tensor += ratio_tensor.clone()
    avg_ratio_tensor /= layer_num

    removed_ratio_tensor = []
    removed_rank_tensor = []

    # divide ratios corresponding to special tokens & punctuations
    for i in [cls_id, sep_id, period_id, comma_id]:
        removed_ratio_tensor.append(avg_ratio_tensor[ids==i])
        removed_rank_tensor.append(rank[ids==i])
        avg_ratio_tensor = avg_ratio_tensor[ids != i]
        rank = rank[ids != i]
        ids = ids[ids != i]

    plt.figure(figsize=(10,7.5), dpi=200)
    plt.scatter(removed_rank_tensor[0], removed_ratio_tensor[0], s=1, alpha=0.05,label="[CLS]", color = cmap(1))
    plt.scatter(removed_rank_tensor[1], removed_ratio_tensor[1], s=1, alpha=0.05,label="[SEP]", color = cmap(2))
    plt.scatter(removed_rank_tensor[2], removed_ratio_tensor[2], s=1, alpha=0.05,label="period", color = cmap(3))
    plt.scatter(removed_rank_tensor[3], removed_ratio_tensor[3], s=1, alpha=0.05,label="comma", color = cmap(4))
    plt.scatter(rank, avg_ratio_tensor, s=1, alpha=0.05,label="other", color = cmap(0))
    leg = plt.legend(fontsize=fs2, frameon=True)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
        lh.set_sizes([100])
    
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.xticks([0,10,100,1000,10000])
    if size in ("base", "large"):
        plt.yticks([0.10, 0.15, 0.20, 0.25])
    elif size == "medium":
        plt.yticks([0.15, 0.2, 0.25, 0.3])
    elif size == "small":
        plt.yticks([0.20, 0.3, 0.4, 0.5])
    elif size == "mini":
        plt.yticks([0.20, 0.25, 0.30, 0.35, 0.4])
    elif size == "tiny":
        plt.yticks([0.30, 0.35, 0.40, 0.45, 0.5])

    plt.xlim(0.8,)
    plt.xscale('log')
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', pad=10)
    plt.xlabel('Frequency rank',fontsize=fs,labelpad=7)
    plt.ylabel('Mixing ratio',fontsize=fs,labelpad=7)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    
    if not os.path.exists(f"./results/relation_with_freq/bert_{size}"):
        os.makedirs(f"./results/relation_with_freq/bert_{size}")
    
    plt.savefig(f"./results/relation_with_freq/bert_{size}/rank_and_ratio.png")
    plt.show()