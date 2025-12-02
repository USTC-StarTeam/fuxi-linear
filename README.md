# FuXi-Linear

This is the  Pytorch implementation for our paper `FuXi-Linear: Unleashing the Power of Linear Attention in Long-term Time-aware Sequential Recommendation`

## Getting started

### Public experiments

To replicate the public experiments conducted in the traditional time-aware sequential recommender setting on Kuairand-27K as described in the paper, please follow these steps:

#### Install dependencies.

Install PyTorch based on official instructions. Then,

```
pip3 install gin-config absl-py scikit-learn scipy matplotlib numpy apex hypothesis pandas fbgemm_gpu iopath
```

#### Download and preprocess data.

Create a directory named `tmp/`.

Visit [https://kuairand.com/](https://kuairand.com/) and [https://kuairec.com/](https://kuairec.com/) to download the respective datasets. Extract the downloaded datasets into the directories `tmp/kuairand-27k` and `tmp/kuairec`.

Next, execute the following commands to preprocess the data:

```bash
python3 preprocess_public_data.py 
python3 preprocess_kuairand27k_data.py
python3 preprocess_kuairec_data.py
```

These instructions will guide you to download the MovieLens-20M dataset and preprocess each of the datasets separately.

#### Run model training.

```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --gin_config_file=configs/kuairand-27k/linear-4b-l1024-b64x2.gin --master_port=12345
```

Other configurations are included in configs/kuairand-27k, configs/ml-20m, configs/kuairec to make reproducing these experiments easier.

#### Verify results.

By default we write experimental logs to exps/. We can launch tensorboard with something like the following:

```
tensorboard --logdir ./exps/kuairand-27k-l1024/ --port 24001 --bind_all
```


