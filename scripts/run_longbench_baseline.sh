# models="Llama-2-7B-32K-Instruct Llama-3-8B-Instruct-Gradient-1048k"
models="Llama-3-8B-Instruct-Gradient-1048k"
# models="Llama-2-7B-32K-Instruct"
# models="Mistral-7B-Instruct-v0.2"

# tasks="samsum narrativeqa qasper triviaqa hotpotqa multifieldqa_en multifieldqa_zh 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"
# Rebuttal - 250425 #
tasks="hotpotqa"

for model in $models; do
    for task in $tasks; do
        bash scripts/longbench_baseline.sh $model $task 
    done
done
