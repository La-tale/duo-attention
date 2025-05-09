model=$1
task=$2
python -u eval/LongBench/pred.py \
    --model $model --task $task \
    --method full
