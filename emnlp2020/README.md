# Norm-based Analysis of Transformer

Code for [Kobayashi+20 Attention is Not Only a Weight: Analyzing Transformers with Vector Norms (EMNLP 2020)](https://www.aclweb.org/anthology/2020.emnlp-main.574/).  
Using this repository, 
* In BERT and Transformer model, you can use vector norms <img src="https://latex.codecogs.com/gif.latex?\left&space;\|&space;\alpha&space;f\left(\mathit{\textbf{x}}&space;\right&space;)&space;\right&space;\|"/> in a format similar to attention weight <img src="https://latex.codecogs.com/gif.latex?\alpha"/>. (see [Quick Start](#Quick-Start))  
* You can reproduce our experiments. (see [Reproducing our experiments](#Reproducing-our-experiments))


## Setup

First, we recommend preparing a new virtual environment (python>=3.6) before executing these commands.  
Install necessary packages: 
    
    $ cd configs
    $ pip install pip-tools
    $ pip-compile requirements.in
    $ pip-sync  

Besides, maybe you would need to reinstall the PyTorch version that suits your environment (see [PyTorch HP](https://pytorch.org/)).  

## Quick Start
This is a quick introduction to use changed libraries ([*transformers*](https://github.com/huggingface/transformers) and [*fairseq*](https://github.com/pytorch/fairseq)).  
Using them, you can use vector norms <img src="https://latex.codecogs.com/gif.latex?\left&space;\|&space;\alpha&space;f\left(\mathit{\textbf{x}}&space;\right&space;)&space;\right&space;\|"/> in a format similar to attention weight <img src="https://latex.codecogs.com/gif.latex?\alpha"/>. (only in BERT and Transformer model).

If you want to use vector norms in a Transformer model, see [changed_fairseq_usage.ipynb](changed_fairseq_usage.ipynb)  
If you want to use vector norms in BERT, see [changed_transformers_usage.ipynb](changed_transformers_usage.ipynb)


## Reproducing our experiments
You can reproduce experiments in [our paper](https://www.aclweb.org/anthology/2020.emnlp-main.574/).  
There are two experiments:  
* Analysis of BERT (Section 4)
* Analysis of Transformer NMT model (Section 5)

If you want to reproduce BERT analysis, see [exp1_bert](exp1_bert).  
If you want to reproduce Transformer NMT model analysis, see [exp2_nmt](exp2_nmt).

## Citation
If you use our code for academic work, please cite:

```
@inproceedings{kobayashi-etal-2020-attention,  
   title = {Attention is Not Only a Weight: Analyzing Transformers with Vector Norms},  
   author = {Kobayashi, Goro and Kuribayashi, Tatsuki and Yokoi, Sho and Inui, Kentaro},  
   booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},  
   year = "2020",  
   url = "https://www.aclweb.org/anthology/2020.emnlp-main.574",  
   pages = "7057--7075",  
}
```
