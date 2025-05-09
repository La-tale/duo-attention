model=$1
task=$2
cluster_size=$3
python -u eval/LongBench/pred_static_cluster.py \
    --model $model --task $task \
    --method cluster \
    --cluster_size ${cluster_size}
