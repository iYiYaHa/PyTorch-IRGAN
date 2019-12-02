# PyTorch-IRGAN


## Description

This project contains a pytorch version implementation about the item recommendation part of [IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models](https://arxiv.org/abs/1705.10513). The official implementation can be found at https://github.com/geek-ai/irgan. If you have any problems on this implementation, please open an issue.


## Requirements

Please refer to requirements.txt

## Project Structure 

.
├── config.py (Configurations about IRGAN and BPR.)
├── data (Data Files)
│   ├── movielens-100k-test.txt
│   └── movielens-100k-train.txt
├── data_utils.py (Utilities about dataset.)
├── evaluation (Evaluation metrics and tools.)
│   ├── __init__.py
│   ├── rank_metrics.py
│   └── rec_evaluator.py
├── exp_notebooks (Notebooks containing experiments for comparison. dns means using pre-trained models with dynamic negative sampling. 
│    │         gen means pre-training generator while dis means pre-training discriminator. SGD and Adam are optimizers adopted.)
│   ├── BPR.ipynb
│   ├── IRGAN-Adam-dns-gen-dis.ipynb
│   ├── IRGAN-Adam-dns-gen.ipynb
│   ├── IRGAN-Adam-without-pretrained-model.ipynb
│   ├── IRGAN-dns-gen-Adam-G-SGD-D.ipynb
│   ├── IRGAN-SGD-dns-gen-dis.ipynb
│   ├── IRGAN-SGD-dns-gen-static-negative-sampling.ipynb
│   ├── IRGAN-SGD-without-pretrained-model.ipynb
│   ├── Pretrain-Discriminator-Dynamic-Negative-Sampling-Adam.ipynb
│   ├── Pretrain-Discriminator-Dynamic-Negative-Sampling.ipynb
│   └── Pretrain-Discriminator-Static-Negative-Sampling.ipynb
├── IRGAN-SGD-dns-gen.ipynb(IRGAN with the SGD optimizer and a pre-trained model with dynamic negative sampling for generator.)
├── model.py (Model definition.)
├── pretrained_models (Pre-trained models)
│   ├── pretrained_model_dns.pkl
│   └── pretrained_model_sns.pkl
├── readme.md

## How to run
1. Execute **conda create --name <env_name> --file requirements.txt** to create an virtual environment and install required packages.
2. Run a jupyter notebook server by **jupyter notebook**.
3. Open IRGAN-SGD-dns-gen.ipynb in a browser and run all cells. 
   The output of loss and other evaluation metrics can be observed with tensorboard.(Other notebooks from the *exp_notebooks* directory can be moved out to its upper-level directory and run). 