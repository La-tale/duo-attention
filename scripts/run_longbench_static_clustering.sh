# models="Llama-2-7B-32K-Instruct Llama-3-8B-Instruct-Gradient-1048k"
models="Llama-2-7B-32K-Instruct"


cluster_sizes="256 512 1024 2048 4096"

# tasks="samsum narrativeqa qasper triviaqa hotpotqa multifieldqa_en multifieldqa_zh 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"
tasks="samsum"

for model in $models; do
    for task in $tasks; do
        for cluster_size in $cluster_sizes; do
            bash scripts/longbench_static_clustering.sh $model $task $cluster_size
        done
    done
done
