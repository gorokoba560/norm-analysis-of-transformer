# Norm-based Analysis of Transformer

Code for [Kobayashi+21 Incorporating Residual and Normalization Layers into Analysis of Masked Language Models (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.373/).  
Using this repository, 
* In BERT, you can use vector norms considering residual connection and layer normalization in addition to attention, in a format similar to attention weigh. (see [Quick Start](#Quick-Start))  
* You can reproduce our experiments. (see [Reproducing our experiments](#Reproducing-our-experiments))


## Setup

First, we recommend preparing a new virtual environment (python>=3.7) before executing these commands.  
Install necessary packages: 
    
    $ cd configs
    $ pip install pip-tools
    $ pip-compile requirements.in
    $ pip-sync  

Besides, install the PyTorch version that suits your environment (see [PyTorch HP](https://pytorch.org/)).  

## Quick Start
This is a quick introduction to use changed library ([*transformers*](https://github.com/huggingface/transformers)).  
Using it, you can use vector norms in a format similar to attention weight. (only in BERT now).

If you want to use vector norms in BERT, see [modified_transformers_usage.ipynb](modified_transformers_usage.ipynb)


## Reproducing our experiments
You can reproduce experiments in [our paper](https://aclanthology.org/2021.emnlp-main.373/).  
See [run.ipynb](run.ipynb)

## Citation
If you use our code for academic work, please cite:
  
```
@inproceedings{kobayashi-etal-2021-incorporating,  
    title = "{I}ncorporating {R}esidual and {N}ormalization {L}ayers into {A}nalysis of {M}asked {L}anguage {M}odels",  
    author = "Kobayashi, Goro  and  
      Kuribayashi, Tatsuki  and  
      Yokoi, Sho  and  
      Inui, Kentaro",  
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",  
    year = "2021",  
    address = "Online and Punta Cana, Dominican Republic",  
    publisher = "Association for Computational Linguistics",  
    url = "https://aclanthology.org/2021.emnlp-main.373",  
    pages = "4547--4568",  
}
```
