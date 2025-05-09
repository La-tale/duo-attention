model=$1
task=$2
attn_pattern=$3
sparsity=$4
python -u attn_score_profile/pred.py \
    --model $model --task $task \
    --method full \
    --attn_load_dir ${attn_pattern} \
    --sparsity $sparsity \
    --sink_size 64 \
    --recent_size 256
# --method duo_attn \
