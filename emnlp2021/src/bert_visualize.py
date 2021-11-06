import torch
from transformers import BertTokenizer, BertModel
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class MixingViz:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def visualize(self, sent1, sent2=None, axes=True, size=30):
        if axes:
            xticklabels = True
            yticklabels = True
        else:
            xticklabels = False
            yticklabels = False
        
        if sent2==None:
            pt_batch = self.tokenizer(sent1, return_tensors="pt").to(self.device)
        else:
            pt_batch = self.tokenizer(sent1, sent2, return_tensors="pt").to(self.device)

        input_ids = pt_batch["input_ids"]
        tokenized_text = self.tokenizer.convert_ids_to_tokens(input_ids[input_ids>0])
        
        with torch.no_grad():
            _, _, attentions, norms = self.model(**pt_batch, output_attentions=True, output_norms=True)

        plt.rcParams['axes.titlesize'] = size
        
        print("Attn-W:")
        fig, axe = plt.subplots(3, 4, figsize=(15, 12), dpi=200)
        for layer in range(12):
            attention = attentions[layer][0].mean(0).cpu().numpy()
            df = pd.DataFrame(attention,columns=tokenized_text,index=tokenized_text)
            sns.heatmap(df,
                        cmap="Reds",
                        square=True, 
                        cbar=False, 
                        ax=axe[int(layer/4)][layer-int(layer/4)*4], 
                        xticklabels=xticklabels, 
                        yticklabels=yticklabels
                       )
            axe[int(layer/4)][layer-int(layer/4)*4].title.set_text(f'Layer {layer+1}')
        plt.savefig(f"./results/example/attn_w.png")
        plt.show()
        
        print("AttnRes-W:")
        fig, axe = plt.subplots(3, 4, figsize=(15, 12), dpi=200)
        for layer in range(12):
            attention = attentions[layer][0].mean(0).cpu().numpy()
            res = np.zeros((len(attention), len(attention)), int)
            np.fill_diagonal(res, 1)
            abnar = 0.5*attention + 0.5*res
            df = pd.DataFrame(abnar,columns=tokenized_text,index=tokenized_text)
            sns.heatmap(df,
                        cmap="Reds",
                        square=True, 
                        cbar=False, 
                        ax=axe[int(layer/4)][layer-int(layer/4)*4], 
                        xticklabels=xticklabels, 
                        yticklabels=yticklabels
                       )
            axe[int(layer/4)][layer-int(layer/4)*4].title.set_text(f'Layer {layer+1}')
        plt.savefig(f"./results/example/attnres_w.png")
        plt.show()
        
        print("Attn-N:")
        fig, axe = plt.subplots(3, 4, figsize=(15, 12), dpi=200)
        for layer in range(12):
            res_summed_afx_norm = norms[layer][1]
            norm = res_summed_afx_norm[0].cpu().numpy()
            df = pd.DataFrame(norm,columns=tokenized_text,index=tokenized_text)
            sns.heatmap(df,
                        cmap="Reds",
                        square=True, 
                        cbar=False, 
                        ax=axe[int(layer/4)][layer-int(layer/4)*4], 
                        xticklabels=xticklabels, 
                        yticklabels=yticklabels
                       )
            axe[int(layer/4)][layer-int(layer/4)*4].title.set_text(f'Layer {layer+1}')
        plt.savefig(f"./results/example/attn_n.png")
        plt.show()
        
        
        print("AttnResLn-N (Proposed):")
        fig, axe = plt.subplots(3, 4, figsize=(15, 12), dpi=200)
        for layer in range(12):
            post_ln_norm = norms[layer][3]
            norm = post_ln_norm[0].cpu().numpy()
            df = pd.DataFrame(norm,columns=tokenized_text,index=tokenized_text)
            sns.heatmap(df,
                        cmap="Reds",
                        square=True, 
                        cbar=False, 
                        ax=axe[int(layer/4)][layer-int(layer/4)*4], 
                        xticklabels=xticklabels, 
                        yticklabels=yticklabels
                       )
            axe[int(layer/4)][layer-int(layer/4)*4].title.set_text(f'Layer {layer+1}')
        plt.savefig(f"./results/example/attnresln_n.png")
        plt.show()

class MixingVizLayer:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def visualize(self, sent1, sent2=None, axes=True, size=30):
        if axes:
            xticklabels = True
            yticklabels = True
        else:
            xticklabels = False
            yticklabels = False
        
        if sent2==None:
            pt_batch = self.tokenizer(sent1, return_tensors="pt").to(self.device)
        else:
            pt_batch = self.tokenizer(sent1, sent2, return_tensors="pt").to(self.device)
    
        input_ids = pt_batch["input_ids"]
        tokenized_text = self.tokenizer.convert_ids_to_tokens(input_ids[input_ids>0])
        
        with torch.no_grad():
            _, _, attentions, norms = self.model(**pt_batch, output_attentions=True, output_norms=True)
        
        print("Attn-W:")
        for layer in range(12):
            plt.figure(figsize=(15, 12), dpi=200)
            attention = attentions[layer][0].mean(0).cpu().numpy()
            df = pd.DataFrame(attention,columns=tokenized_text,index=tokenized_text)
            sns.heatmap(df,
                        cmap="Reds",
                        square=True, 
                        cbar=False, 
                        xticklabels=xticklabels, 
                        yticklabels=yticklabels
                       )
        plt.savefig(f"./results/example/attn_w_layer{layer+1}.png")
        plt.show()
        
        print("AttnRes-W:")
        for layer in range(12):
            plt.figure(figsize=(15, 12), dpi=200)
            attention = attentions[layer][0].mean(0).cpu().numpy()
            res = np.zeros((len(attention), len(attention)), int)
            np.fill_diagonal(res, 1)
            abnar = 0.5*attention + 0.5*res
            df = pd.DataFrame(abnar,columns=tokenized_text,index=tokenized_text)
            sns.heatmap(df,
                        cmap="Reds",
                        square=True, 
                        cbar=False, 
                        xticklabels=xticklabels, 
                        yticklabels=yticklabels
                       )
        plt.savefig(f"./results/example/attnres_w_layer{layer+1}.png")
        plt.show()
        
        print("Attn-N:")
        for layer in range(12):
            plt.figure(figsize=(15, 12), dpi=200)
            res_summed_afx_norm = norms[layer][1]
            norm = res_summed_afx_norm[0].cpu().numpy()
            df = pd.DataFrame(norm,columns=tokenized_text,index=tokenized_text)
            sns.heatmap(df,
                        cmap="Reds",
                        square=True, 
                        cbar=False, 
                        xticklabels=xticklabels, 
                        yticklabels=yticklabels
                       )
        plt.savefig(f"./results/example/attn_n_layer{layer+1}.png")
        plt.show()
        
        
        print("AttnResLn-N (Proposed):")
        for layer in range(12):
            plt.figure(figsize=(15, 12), dpi=200)
            post_ln_norm = norms[layer][3]
            norm = post_ln_norm[0].cpu().numpy()
            df = pd.DataFrame(norm,columns=tokenized_text,index=tokenized_text)
            sns.heatmap(df,
                        cmap="Reds",
                        square=True, 
                        cbar=False, 
                        xticklabels=xticklabels, 
                        yticklabels=yticklabels
                       )
        plt.savefig(f"./results/example/attnresln_n_layer{layer+1}.png")
        plt.show()