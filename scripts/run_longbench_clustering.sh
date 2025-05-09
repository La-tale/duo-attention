# models="Llama-2-7B-32K-Instruct Llama-3-8B-Instruct-Gradient-1048k"
models="Llama-2-7B-32K-Instruct"


## Total budget: y ##
## 2*centroid_budget = 2*y - 0.275 --> centroid_budget = y - 0.1375 (where y = 0.3, 0.4, ... 1)
budgets="0.0625 0.1625 0.2625 0.3625 0.4625 0.5625 0.6625 0.7625 0.8625"
# budgets="0.5625 0.6625 0.7625 0.8625"
# budgets="0.1625"

# tasks="samsum narrativeqa qasper triviaqa hotpotqa multifieldqa_en multifieldqa_zh 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"
# tasks="samsum"
tasks="narrativeqa qasper triviaqa hotpotqa multifieldqa_en multifieldqa_zh 2wikimqa musique dureader gov_report qmsum multi_news vcsum trec lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"

for model in $models; do
    for task in $tasks; do
        for budget_ratio in $budgets; do
            # 파일 경로 설정
            file_path="eval/LongBench/config/budget_${budget_ratio}.json"

            # 파일이 이미 존재하는지 확인
            if [ -e "$file_path" ]; then
                echo "File $file_path already exists. Skipping file creation."
            else
                # JSON 내용 작성
                echo "{\"budget_ratio\": $budget_ratio}" > $file_path
                echo "Config file created at $file_path"
            fi

            bash scripts/longbench_clustering.sh $model $task $file_path $budget_ratio
        done
    done
done
