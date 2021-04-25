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
dgl
sklearn
allennlp
```



# Data 

The current data directory only includes the sample data. 
ACE05 Datset requires LDC License ([Access from LDC](https://catalog.ldc.upenn.edu/LDC2006T06) and preprocessing following [OneIE](http://blender.cs.illinois.edu/software/oneie/). 

You may contact `qizeng2@illinois.edu` for the preprocessed (enhanced) data.

# Train

```
CUDA_VISIBLE_DEVICES=0 python main.py --mode 'train' --version 'test' --model_base 'SEM_ARC' 
```

- `mode`: train, infer, eval
- `model_base`:  SEM_ARC, SEM, ARC, SKG, DGI
- `version`: name for this model

Check `args.py` for more tunable hyperparameters. 

# Eval

The evaluation code of **Event Coreference** can be found in `event-coref` folder with a separate README.

The evaluation for **Node Typing** and **Argument Role Classification** can be run with:
```
CUDA_VISIBLE_DEVICES=0 python main_hetero.py --mode 'eval' --version 'test' --load_emb 'SEM_ARC.test'
```
- `load_emb`: [MODELBASE].[VERSION]
