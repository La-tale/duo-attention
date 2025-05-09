#!/bin/bash

# Create and activate conda environment
# conda create -yn duo python=3.10 -y
# source activate duo
#
# # Install necessary packages
conda install -y git
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit
conda install -y nvidia::cuda-cudart-dev
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install transformers accelerate sentencepiece datasets wandb zstandard matplotlib huggingface_hub
pip install tensor_parallel
pip install ninja packaging
pip install flash-attn --no-build-isolation

# LongBench evaluation
pip install seaborn rouge_score einops pandas

pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install DuoAttention
pip install -e .

# Install Block Sparse Streaming Attention
# git clone git@github.com:mit-han-lab/Block-Sparse-Attention.git
# cd Block-Sparse-Attention
# python setup.py install

