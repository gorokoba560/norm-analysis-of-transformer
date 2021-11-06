# Norm-based Analysis of Transformer

Implementations for 2 papers introducing to analyze Transformers using vector norms:
* [Kobayashi+'20 Attention is Not Only a Weight: Analyzing Transformers with Vector Norms (EMNLP 2020)](https://www.aclweb.org/anthology/2020.emnlp-main.574/)
* [Kobayashi+'21 Incorporating Residual and Normalization Layers into Analysis of Masked Language Models (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.373/)


## [Kobayashi+'20 Attention is Not Only a Weight: Analyzing Transformers with Vector Norms (EMNLP 2020)](https://www.aclweb.org/anthology/2020.emnlp-main.574/)
This paper proposed to analyze attention, a core component of Transformer, using vector norms rather than attention weights.  
Transformer analyses have been focused on mixing in attention and have typically observed attention weights.  
However, in addition to attention weights, there are more factors to determine attention's outputs: the input vector itself and vector transformations.  
Then, this paper proposed to analyze attention using vector norms considering them.  
 → Check this paper's code: [Code for emnlp2020](emnlp2020).  


## [Kobayashi+'21 Incorporating Residual and Normalization Layers into Analysis of Masked Language Models (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.373/)
This paper proposed to analyze attention block (i.e., attention, residual connection, and layer normalization) using vector norms.  
Transformer analyses have been focused on mixing in attention.  
However, there are components other than attention in Transformer, and they can play a role other than mixing.  
Then, this paper proposed to expand the scope of Transformer analysis from attention into attention block.  
 → Check this paper's code: [Code for emnlp2021](emnlp2021).  


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
@inproceedings{kobayashi-etal-2021-incorporating,
   title = {Incorporating Residual and Normalization Layers into Analysis of Masked Language Models},
   author = {Kobayashi, Goro and Kuribayashi, Tatsuki and Yokoi, Sho and Inui, Kentaro},
   booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Proceeding (EMNLP)},
   year = "2021",
   url = "https://arxiv.org/abs/2109.07152",
   pages = "to appear",
}
```
