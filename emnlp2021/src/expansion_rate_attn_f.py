import torch
from transformers import BertModel, BertForPreTraining, BertConfig, RobertaModel
import numpy as np
import matplotlib.pyplot as plt
import os
import math


def expansion_rate(model_name="bert-base-uncased"):
    value_weight = ()
    value_bias = ()
    dense_weight = ()
    dense_bias = ()

    model = BertModel.from_pretrained(model_name)
    hidden_size = model.config.hidden_size

    singular_values_lis = []

    for (name, param) in model.named_parameters():
        if name.endswith(".attention.output.dense.weight"):
            dense_weight = dense_weight + (param,)
        if name.endswith(".attention.output.dense.bias"):
            dense_bias = dense_bias + (param,)
        if name.endswith("value.weight"):
            value_weight = value_weight + (param,)
        if name.endswith("value.bias"):
            value_bias = value_bias + (param,)

    for l in range(model.config.num_hidden_layers):
        zeros = torch.zeros((1,hidden_size+1))
        
        value = torch.cat((value_weight[l], torch.unsqueeze(value_bias[l],1)),dim=1)
        dense = torch.cat((torch.transpose(dense_weight[l], 0, 1), torch.unsqueeze(dense_bias[l],1)),dim=1)
        value = torch.cat((value, zeros), dim=0)
        dense = torch.cat((dense, zeros), dim=0)
        value[-1][-1] = 1
        dense[-1][-1] = 1

        transform = value.t().matmul(dense)
        u, s, v = torch.svd(transform) 
        singular_values = s.detach().numpy()
        singular_values_lis.append(singular_values)

    expans_rate_mean = 0
    print("Layer\t\t Mean")
    print("-"*30)
    for l in range(model.config.num_hidden_layers):
        expans_rate = 0
        for s in singular_values_lis[l]:
            expans_rate += s**2
        expans_rate = math.sqrt(expans_rate) / math.sqrt(hidden_size+1)
        expans_rate_mean += expans_rate
        print(f"Layer {l+1} :\t{expans_rate:.3f}")
    expans_rate_mean /= model.config.num_hidden_layers
    print(f"Mean :\t\t{expans_rate_mean:.3f}")
    print("-"*30)