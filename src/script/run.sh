torun_file="/public/home/hongy/zpwang/LLM_Reasoning/src/script/pred_subtext_chinese.py"
torun_file="/public/home/hongy/zpwang/LLM_Reasoning/src/script/pred_base_IICOT.py"
torun_file="/public/home/hongy/zpwang/LLM_Reasoning/src/script/pred_base.py"
torun_file="/public/home/hongy/zpwang/LLM_Reasoning/src/script/pred_subtext.py"
torun_file="/public/home/hongy/zpwang/LLM_Reasoning/src/script/pred_subtext_judge.py"
torun_file="/public/home/hongy/zpwang/LLM_Reasoning/src/script/pred_base_fewshot.py"
torun_file="/public/home/hongy/zpwang/LLM_Reasoning/src/script/train_subtext.py"
torun_file="/public/home/hongy/zpwang/LLM_Reasoning/src/script/dev_subtext.py"

start_time=$(date +%Y-%m-%d-%H-%M-%S)
filename=$(basename "$torun_file")
filename="${filename%.*}"

nohup /public/home/hongy/miniconda3/envs/zpwang_main/bin/python $torun_file > "/public/home/hongy/zpwang/LLM_Reasoning/logs/${start_time}.${filename}.log" 2>&1 &
