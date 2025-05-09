# Original code 
pred.py: revised version of original "pred.py" to support multi-gpu
pred\_multi\_gpu.py: pred.py + decoding\_simulation\_length=0

# Code for evaluating clustering
pred\_cluster.py: evaluation code for clustering with budget ratio (0~1 / e.g., 0.25, 0.5, ...)
pred\_static\_cluster.py: evaluation code for clustering with static budget (e.g., 256 tokens, 512 tokens ...)
