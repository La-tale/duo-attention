model=$1
task=$2
cluster_args_file=$3
budget=$4
python -u eval/LongBench/pred_cluster.py \
    --model $model --task $task \
    --method cluster \
    --cluster_args_path ${cluster_args_file} \
    --budget $budget
