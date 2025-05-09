#!/bin/bash

# MODEL_LIST=("togethercomputer/LLaMA-2-7B-32K" "gradientai/Llama-3-8B-Instruct-Gradient-1048k")
# MODEL_LIST=("togethercomputer/LLaMA-2-7B-32K")
MODEL_LIST=("gradientai/Llama-3-8B-Instruct-Gradient-1048k")

# 각 모델에 대한 prefill_length와 gen_length 설정 (KB 단위)
# declare -A SETTINGS
# SETTINGS["togethercomputer/LLaMA-2-7B-32K"]="((8*1024 24*1024) (16*1024 16*1024) (24*1024 8*1024))"
# SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]="((256*1024 768*1024) (512*1024 512*1024) (768*1024 256*1024))"


# MODEL_LIST=("togethercomputer/LLaMA-2-7B-32K")

# 각 모델에 대한 prefill_length와 gen_length 설정 (KB 단위)
declare -A SETTINGS
# SETTINGS["togethercomputer/LLaMA-2-7B-32K"]="16*1024 16*1024"
# SETTINGS["togethercomputer/LLaMA-2-7B-32K"]="8*1024 12*1024 16*1024 20*1024 24*1024"
# SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]="256*1024 768*1024 512*1024 512*1024 768*1024 256*1024"
# SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]="128*1024 896*1024"
# SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]="32*1024 64*1024 128*1024 256*1024 512*1024 1024*1024"
# SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]="32*1024 64*1024 128*1024 256*1024 512*1024 1024*1024"
# SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]="128*1024 256*1024 512*1024 1024*1024"
# SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]="128*1024"
#
# Many samples #
SETTINGS["togethercomputer/LLaMA-2-7B-32K"]=""
SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]=""

for i in $(seq 8 2 24); do
    SETTINGS["togethercomputer/LLaMA-2-7B-32K"]="${SETTINGS["togethercomputer/LLaMA-2-7B-32K"]}${i}*1024 "
done

for i in $(seq 256 16 1024); do
    SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]="${SETTINGS["gradientai/Llama-3-8B-Instruct-Gradient-1048k"]}${i}*1024 "
done

cache_type="OffloadedClusteringCache_wo_norm"

for model in "${MODEL_LIST[@]}"
do
    # 모델 이름에서 슬래시를 언더스코어로 변경하여 출력 디렉토리용 변수 생성
    model_name=$(echo "$model" | tr '/' '_')
    
    # 해당 모델의 설정 가져오기
    settings=(${SETTINGS[$model]})
    
    # 설정을 쌍으로 처리하기 위해 인덱스 사용
    for ((i=0; i<${#settings[@]}; i+=1))
    do
        ctx_length=$((${settings[$i]}))
        echo "Decoding context length: $ctx_length"

        # # Python 스크립트 실행
        python sms_workspace/benchmark_offloaded_proposed_clustering_decode_wo_overhead_only.py \
            --model_name "$model" \
            --max_length "$ctx_length" \
            --output_dir "sms_outputs/paper_efficiency/${model_name}/motiv_decode_proposed_cluster_wo_overhead_only_${cache_type}__gen${ctx_length}_64cpu_core" \
            --seed 42 \
            --cache_type ${cache_type} \
            --prefilling_chunk_size 32768
    done
done
