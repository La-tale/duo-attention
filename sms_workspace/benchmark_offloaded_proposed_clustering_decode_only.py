import torch
import os
import argparse
import sys


from duo_attn.utils import (
    #  get_model,
    #  get_tokenizer,
    #  parse_args,
    to_device,
    seed_everything,
)

from utils import bench_func

import transformers
from transformers.cache_utils import OffloadedCache, DynamicCache
from patch.llama_clustering import enable_proposed_clustering
from patch.ClusterCache import OffloadedClusteringCache, DynamicClusteringCache, OffloadedClusteringCache_wo_norm

debug=0

def parse_args():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Benchmark offload execution script.")

    # Add arguments
    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the model to use."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        required=True,
        help="Maximum sequence length for generation."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the output results."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--prefilling_chunk_size",
        type=int,
        required=True,
        help="Chunk size for prefilling (e.g., context length)."
    )
    parser.add_argument(
        "--cache_type",
        required=True,
        help="DynamicCache - Ideal, OffloadedCache - Baseline"
    )

    args = parser.parse_args()
    # Parse arguments
    return args

def get_model_and_tokenizer(model_name):
    #  config = transformers.AutoConfig.from_pretrained(model_name)
    #  config.vocab_size = 4096
    #  breakpoint()
    ## Same as duo except attn_implementation=" and low_cpu_mem_usage
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        #  torch_dtype=torch.bfloat16,
        #  low_cpu_mem_usage=True,
        #  attn_implementation="eager", ## Need to implement more
        attn_implementation="flash_attention_2",
    )

    ## Same as duo
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, use_fast=False, trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    return model, tokenizer

if __name__ == "__main__":
    args = parse_args() 

    if args.seed is not None:
        seed_everything(args.seed)

    with torch.no_grad():
        model, tokenizer = get_model_and_tokenizer(args.model_name)
    
    model.eval()
    
    ## SMS' COMMENTS ## . 2025-01-25
    ## To do - Argumentation ##
    #  if args.cluster_args_path is not None:
    #      assert args.cache_type == "OffloadedCache", "You should use OffloadedCache when using budget config"
    #      cluster_args = json.load(open(args.cluster_args_path, "r"))
    #      enable_clustering(model, cluster_args)
    enable_proposed_clustering(model)

    ## SMS' COMMENTS ## . 2025-03-26
    ## For multi-gpu env ##
    pp=False
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            pp=True
    model = to_device(model, "cuda", enable_pp=pp)

    text = "a\n\n" * args.max_length

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")[
        :, : args.max_length - 1
    ]

    print(input_ids.shape)

    if args.cache_type == "DynamicCache":
        cache_type=DynamicCache
    elif args.cache_type == "OffloadedCache":
        cache_type=OffloadedCache
    elif args.cache_type == "OffloadedClusteringCache":
        cache_type=OffloadedClusteringCache
    elif args.cache_type == "DynamicClusteringCache":
        cache_type=DynamicClusteringCache
    elif args.cache_type == "OffloadedClusteringCache_wo_norm":
        cache_type=OffloadedClusteringCache_wo_norm

    prefilling_chunk_size = args.prefilling_chunk_size

    kv_cache = cache_type()
    with torch.no_grad():
        for i in range(0, input_ids.size(1), prefilling_chunk_size):
            input_chunk = input_ids[:, i : i + prefilling_chunk_size]
            outputs = model(
                input_ids=input_chunk,
                past_key_values=kv_cache,
                use_cache=True,
            )
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    

    def func2():
        with torch.no_grad():
            _ = model(
                input_ids=pred_token_idx,
                past_key_values=kv_cache,
                use_cache=True,
            )
        ## For simplicity ##
        ## SMS' COMMENTS ## . 2025-01-07
        kv_cache.crop(-1)
        #  if args.budget is None:
        #      kv_cache.crop(-1)
        #  elif crop_cond:
        #      kv_cache.crop(-1)
        #  kv_cache.evict_last(1)
    
    ## Debug #
    if debug:
        func2()
    else:
        gen_latency, gen_memory = bench_func(func2, num_steps=5, num_warmup_steps=2)

    #  kv_cache_memory_usage = kv_cache.memory_usage / 1024 / 1024
    if debug:
        print(f"Average generation time: {gen_latency:.4f} ms")
        print(f"Peak generation memory usage: {gen_memory:.4f} MB")
        print(f"Average context time: {ctx_latency:.4f} ms")
        print(f"Peak context memory usage: {ctx_memory:.4f} MB")
        print(f"Model name: {args.model_name}")
        print(f"Context length: {args.max_length}")
    elif args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "benchmark_result.txt"), "w") as f:
            print(f"Average generation time: {gen_latency:.4f} ms", file=f)
            print(f"Peak generation memory usage: {gen_memory:.4f} MB", file=f)
            print(f"Model name: {args.model_name}", file=f)
            print(f"Context length: {args.max_length}", file=f)
            print(f"Prefilling chunk size: {prefilling_chunk_size}", file=f)
            #  print(f"KV cache memory usage: {kv_cache_memory_usage:.4f} MB", file=f)
