#!/bin/bash

# MODEL_LIST=("togethercomputer/LLaMA-2-7B-32K" "gradientai/Llama-3-8B-Instruct-Gradient-1048k")
# MODEL_LIST=("togethercomputer/LLaMA-2-7B-32K-Instruct")
MODEL_LIST=("Llama-2-7B-32K-Instruct")
# MODEL_LIST=("gradientai/Llama-3-8B-Instruct-Gradient-1048k")

# 각 모델에 대한 prefill_length와 gen_length 설정 (KB 단위)
# declare -A SETTINGS
# SETTINGS["togethercomputer/LLaMA-2-7B-32K"]="((8*1024 24*1024) (16*1024 16*1024) (24*1024 8*1024))"
# SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]="((256*1024 768*1024) (512*1024 512*1024) (768*1024 256*1024))"


# MODEL_LIST=("togethercomputer/LLaMA-2-7B-32K")

# 각 모델에 대한 prefill_length와 gen_length 설정 (KB 단위)
declare -A SETTINGS
# SETTINGS["togethercomputer/LLaMA-2-7B-32K"]="16*1024 16*1024"
SETTINGS["Llama-2-7B-32K-Instruct"]="8*1024 24*1024 16*1024 16*1024 24*1024 8*1024"
SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]="256*1024 1 512*1024 1 768*1024 1"
# SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]="128*1024 896*1024"
# SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]="32*1024 1*1024"

attn_pattern_name="lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
sparsity="0.6"
cache_type="DuoOffloadedCache"

for model in "${MODEL_LIST[@]}"
do
    # 모델 이름에서 슬래시를 언더스코어로 변경하여 출력 디렉토리용 변수 생성
    model_name=$(echo "$model" | tr '/' '_')
    
    # 해당 모델의 설정 가져오기
    settings=(${SETTINGS[$model]})
    
    # 설정을 쌍으로 처리하기 위해 인덱스 사용
    for ((i=0; i<${#settings[@]}; i+=2))
    do
        prefill_length=$((${settings[$i]}))
        gen_length=$((${settings[$i+1]}))

        echo "Prefil length: $prefill_length"
        echo "Generation length: $gen_length"
        
        # Python 스크립트 실행
        python eval/efficiency/benchmark_offloaded_e2e.py \
            --model_name models/${model} \
            --prefill_length "$prefill_length" \
            --gen_length "$gen_length" \
            --output_dir "sms_outputs/paper_efficiency/${model_name}/${cache_type}__pre${prefill_length}__gen${gen_length}_sparsity_${sparsity}_cm_agpu1" \
            --attn_load_dir "attn_patterns/${model_name}/${attn_pattern_name}" \
            --sparsity $sparsity \
            --seed 42 \
            --prefilling_chunk_size 32768
    done
done
