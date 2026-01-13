# ACLNet

This is the official PyTorch implementation for "Affinity Contrastive Learning for Skeleton-based Human Activity Understanding".

## Installation

```shell
git clone https://github.com/aclnet/ACLNet.git
cd ACLNet
conda env create -f aclnet.yaml
conda activate aclnet
pip install -e .
```

## Data Preparation

PYSKL provides links to the pre-processed skeleton pickle annotations.

- NTU RGB+D: [NTU RGB+D Download Link](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl)
- NTU RGB+D 120: [NTU RGB+D 120 Download Link](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl)
- Kinetics-Skeleton: [Kinetics-Skeleton Download Link](https://download.openmmlab.com/mmaction/pyskl/data/k400/k400_hrnet.pkl)
- PKU-MMD: [PKU-MMD Download Link](https://drive.google.com/file/d/1tuwvGb_F2nSQj7kegfQCyLLk4LTA809O/view?usp=sharing)
- FineGYM: [FineGYM Download Link](https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_hrnet.pkl)


For Kinetics-Skeleton, since the skeleton annotations are large, please use the [Kinetics Annotation Link](https://www.dropbox.com/scl/fi/5phx0m7bok6jkphm724zc/kpfiles.zip?rlkey=sz26ljvlxb6gwqj5m9jvynpg8&st=47vcw2xb&dl=0) to download the `kpfiles` and extract it under `$ACLNet/data/k400` for Kinetics-Skeleton. 

Note that the `kpfiles` needs to be extracted under `Linux`. Additionally, Kinetics-Skeleton requires the dependency `Memcached` to run, which could be referred to [here](https://www.runoob.com/memcached/memcached-install.html). 

You could check the official [Data Doc](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md) of PYSKL for more detailed instructions.

## Training & Testing

Please change the config file depending on what you want. You could use the following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.

```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# For example: train on NTU RGB+D X-Sub (Joint Modality) with 1 GPU, with validation, and test the checkpoint.
bash tools/dist_train.sh configs/ntu60_xsub/j.py 1 --validate --test-last --test-best
```

```shell
# Testing
bash tools/dist_test.sh {config_name} {checkpoint_file} {num_gpus} {other_options}
# For example: test on NTU RGB+D X-Sub (Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/ntu60_xsub/j.py checkpoints/CHECKPOINT.pth 1 --eval top_k_accuracy --out result.pkl
```

```shell
# Ensemble the results
cd tools
python ensemble.py
```

## Pretrained Models

All the checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1UheUd00odocDjtQtWY9SNIbFDuA9qs1a?usp=sharing).

For the detailed performance of pretrained models, please go to the [Model Doc](/data/README.md).

## Acknowledgements

This repo is mainly based on [PYSKL](https://github.com/kennymckormick/pyskl). We also refer to [MS-G3D](https://github.com/kenziyuliu/ms-g3d), [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), and [FR-Head](https://github.com/zhysora/FR-Head).

Thanks to the original authors for their excellent work!

