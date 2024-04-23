start_time=$(date +%Y-%m-%d-%H-%M-%S)
# source activate zpwang_main
# torun_file="/public/home/hongy/zpwang/LLM_Reasoning/src/script/pred_subtext.py"
torun_file="/public/home/hongy/zpwang/LLM_Reasoning/src/script/pred_init.py"
nohup /public/home/hongy/miniconda3/envs/zpwang_main/bin/python $torun_file > "/public/home/hongy/zpwang/LLM_Reasoning/logs/${start_time}.log" 2>&1 &