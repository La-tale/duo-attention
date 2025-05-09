attn_pattern_name="lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
# models="Llama-2-7B-32K-Instruct Llama-3-8B-Instruct-Gradient-1048k"
models="Llama-2-7B-32K-Instruct"

# sparsities="0 0.5 0.75"
sparsities="0"

# tasks="samsum narrativeqa qasper triviaqa hotpotqa multifieldqa_en multifieldqa_zh 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"
tasks="samsum"

for model in $models; do
    for task in $tasks; do
        for sparsity in $sparsities; do
            bash scripts/longbench_score_profile.sh $model $task "attn_patterns/${model}/${attn_pattern_name}" $sparsity
        done
    done
done
