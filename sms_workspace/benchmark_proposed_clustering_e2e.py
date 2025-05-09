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
from patch.llama_clustering import enable_clustering, enable_proposed_clustering
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
        "--prefill_length",
        type=int,
        required=True,
        help="input sequence length for generation."
    )
    parser.add_argument(
        "--gen_length",
        type=int,
        required=True,
        help="to be generated sequence length for generation."
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
    ## SMS' COMMENTS ## . 2025-01-25
    ## To do - Argumentation ##
    #  parser.add_argument(
    #          '--cluster_args_path',
    #          type=str,
    #          default=None,
    #          help="Path to the cluster args"
    #  )

    args = parser.parse_args()
    # Parse arguments
    return args

def get_model_and_tokenizer(model_name):
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
    
    #  model.model.layers=model.model.layers[:8] # Confine # of layer to four
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

    
    ### Generate Problem ###
    #  batch= 1024
    #  n_samples = 4095
    #  feat_dim = 4
    #  X = torch.randn((batch, n_samples, feat_dim), device='cuda')
    #  breakpoint()

    text = "a\n\n" * args.prefill_length

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")[
        :, : args.prefill_length
    ]
    #  breakpoint()
    #  input_ids = torch.randint(1, 29874, input_ids.shape, device=model.device)
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

    def func1():
        with torch.no_grad():
            for i in range(0, input_ids.size(1), prefilling_chunk_size):
                input_chunk = input_ids[:, i : i + prefilling_chunk_size]
                outputs = model(
                    input_ids=input_chunk,
                    past_key_values=cache_type(),
                    use_cache=True,
                )
    ### Debug ###
    if debug:
        func1()
    else:
        ctx_latency, ctx_memory = bench_func(func1, num_steps=3, num_warmup_steps=2)

    def e2e_func():
        ## SMS' COMMENTS ## . 2025-03-24
        ## prefill ##
        with torch.no_grad():
            kv_cache = cache_type()
            for i in range(0, input_ids.size(1), prefilling_chunk_size):
                input_chunk = input_ids[:, i : i + prefilling_chunk_size]
                outputs = model(
                    input_ids=input_chunk,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            for i in range(0, args.gen_length):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    if debug:
        e2e_func()
    else:
        e2e_latency, e2e_memory = bench_func(e2e_func, num_steps=1, num_warmup_steps=0)
    
    if debug:
        #  kv_cache_memory_usage = kv_cache.memory_usage / 1024 / 1024
        print(f"Average generation time: {gen_latency:.4f} ms" )
        print(f"Peak generation memory usage: {gen_memory:.4f} MB" )
        print(f"Average context time: {ctx_latency:.4f} ms" )
        print(f"Peak context memory usage: {ctx_memory:.4f} MB" )
        print(f"Model name: {args.model_name}" )
        print(f"Context length: {args.prefill_length}" )
        print(f"Prefilling chunk size: {prefilling_chunk_size}" )
        #  print(f"KV cache memory usage: {kv_cache_memory_usage:.4f} MB", )
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "benchmark_result.txt"), "w") as f:
            print(f"Average prefill time: {ctx_latency:.4f} ms", file=f)
            print(f"Peak prefill memory usage: {ctx_memory:.4f} MB", file=f)
            print(f"Average e2e time: {e2e_latency:.4f} ms", file=f)
            print(f"Peak e2e memory usage: {e2e_memory:.4f} MB", file=f)
            print(f"Model name: {args.model_name}", file=f)
            print(f"Context length: {args.prefill_length}", file=f)
            print(f"Generation length: {args.gen_length}", file=f)
            #  print(f"KV cache memory usage: {kv_cache_memory_usage:.4f} MB", file=f)
