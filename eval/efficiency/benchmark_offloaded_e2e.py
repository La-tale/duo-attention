import torch
import argparse
import os

import transformers

from duo_attn.utils import (
    get_model,
    get_tokenizer,
    #  parse_args,
    to_device,
    load_attn_pattern,
    seed_everything,
    sparsify_attention_heads,
)
from duo_attn.patch.llama import (
    enable_llama_duo_attention_offloaded_kv_cache_eval,
    DuoAttentionStaticKVCache,
    OffloadedDuoAttentionStaticKVCache
)
from utils import bench_func

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
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--attn_load_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()



    tokenizer = get_tokenizer(args.model_name) ## Same w/ efficiency code for CAL

    with torch.no_grad():
        ## SMS' COMMENTS ## . 2025-05-01
        #  model = get_model(args.model_name)
        model_name = args.model_name
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16, ## Same w/ cluster_code
            #  low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

    model.eval()

    model = to_device(model, "cuda:0")

    if args.attn_load_dir is not None:
        full_attention_heads, sink_size, recent_size = load_attn_pattern(
            args.attn_load_dir
        )

        full_attention_heads, sparsity = sparsify_attention_heads(
            full_attention_heads, None, args.sparsity
        )
        print(f"True Sparsity: {sparsity}")
        enable_llama_duo_attention_offloaded_kv_cache_eval(model, full_attention_heads)

    text = "a\n\n" * args.prefill_length

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")[
        :, : args.prefill_length
    ]

    print(input_ids.shape)

    #  max_size = input_ids.size(1) + 5
    max_size = input_ids.size(1) + args.gen_length
    prefilling_chunk_size = args.prefilling_chunk_size
    print(f"Max size: {max_size}, Prefilling chunk size: {prefilling_chunk_size}")

    kv_cache = OffloadedDuoAttentionStaticKVCache(
        model,
        full_attention_heads,
        1,
        max_size,
        sink_size,
        recent_size,
    )

    # pre-filling
    def func1():
        with torch.no_grad():
            for i in range(0, input_ids.size(1), prefilling_chunk_size):
                input_chunk = input_ids[:, i : i + prefilling_chunk_size]
                outputs = model(
                    input_ids=input_chunk,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
            kv_cache.clear()

    ctx_latency, ctx_memory = bench_func(func1, num_steps=3, num_warmup_steps=2)
            

    kv_cache_e2e = OffloadedDuoAttentionStaticKVCache(
        model,
            full_attention_heads,
            1,
            max_size,
            sink_size,
            recent_size,
        )
    
    def e2e_func():
        with torch.no_grad():
            for i in range(0, input_ids.size(1), prefilling_chunk_size):
                input_chunk = input_ids[:, i : i + prefilling_chunk_size]
                outputs = model(
                    input_ids=input_chunk,
                    past_key_values=kv_cache_e2e,
                    use_cache=True,
                )
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            for i in range(0, args.gen_length):
                #  print(kv_cache[0][0][1].shape)
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=kv_cache_e2e,
                    use_cache=True,
                )
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    e2e_latency, e2e_memory = bench_func(e2e_func, num_steps=1, num_warmup_steps=0)

    if args.output_dir is not None:
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
