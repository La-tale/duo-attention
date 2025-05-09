import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn


from duo_attn.utils import (
    get_model,
    get_tokenizer,
    parse_args,
    to_device,
    load_attn_pattern,
    seed_everything,
    sparsify_attention_heads,
)
from duo_attn.patch.llama import (
    enable_llama_duo_attention_static_kv_cache_eval,
    DuoAttentionStaticKVCache,
)
# 1. 모델 로드 및 GPU 설정
model_name = "./models/Llama-2-7B-32K-Instruct"  # 실제 Llama 모델 이름을 여기에 입력하세요
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.config.output_attentions=True

# 4. 프로파일링을 위한 입력 데이터 준비
input_text = "This is a sample input."
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 5. torch.profiler 설정 및 실행
outputs = model(**inputs)
breakpoint()
