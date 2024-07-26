# ERIC

[NeurIPS-2022] Efficient Graph Similarity Computation with Alignment Regularization

The final version of the code is to be further cleaned.

## Requirements

```
star虚拟环境名==zhj2
python==3.8
torch==2.1.0
torch_scatter==2.1.2
torch_sparse==0.6.18
torch_geometric==2.4.0
```

# Run Pretrained Model

```
python main.py --dataset AIDS700nef --run_pretrain --pretrain_path model_saved/AIDS700nef/2022-03-17_10-42-12
```

```
python main.py --dataset LINUX --run_pretrain --pretrain_path model_saved/LINUX/2022-03-20_03-01-57
```

## Running Example

![image](img/img.png)
