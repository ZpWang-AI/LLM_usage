start_time=$(date +%Y-%m-%d-%H-%M-%S)
source activate zpwang_main
nohup /public/home/hongy/miniconda3/bin/python /public/home/hongy/zpwang/LLM_API/llm_reasoning/generate.py > "/public/home/hongy/zpwang/LLM_API/llm_reasoning/logs/${start_time}.log" 2>&1 &