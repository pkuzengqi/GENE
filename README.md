# GENE

Code for TextGraphs 2021 paper "GENE: Global Event Network Embedding"

```
@inproceedings{zeng-etal-2021-gene,
    title = "GENE: Global Event Network Embedding",
    author = "Zeng, Qi  and
      Li, Manling  and
      Lai, Tuan  and
      Ji, Heng  and
      Bansal, Mohit  and
      Tong, Hanghang",
    booktitle = "Proceedings of the Graph-based Methods for Natural Language Processing (TextGraphs)",
    month = Jun,
    year = "2021",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
}
```
# Requirements

```
numpy
torch
pytorch_pretrained_bert
stanfordnlp  
dgl
allennlp
mxnet
```

If you preprocessing the data, 
download [bert model](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and put under `./bert/`


# Data 

The current data directory only includes the sample data. ACE05 Datset requires LDC License ([Access from LDC](https://catalog.ldc.upenn.edu/LDC2006T06) and preprocessing following [OneIE](http://blender.cs.illinois.edu/software/oneie/). You may contact qizeng2@illinois.edu for the preprocessed (enhanced) data.


# Train

```
CUDA_VISIBLE_DEVICES=0 python main.py --mode 'train' --version 'test' --model_base 'SEM_ARC' 
```

- `mode`: train, infer, eval
- `version`: 
- `model_base`:  


# Eval

The evaluation code of **Event Coreference** can be found in `event-coref` folder with a separate README.

The evaluation for **Node Typing** and **Argument Role Classification** can be run with:
```
CUDA_VISIBLE_DEVICES=1 python main_hetero.py --mode 'eval' --version 'test' --load_emb 'SEM_ARC.test'
```

