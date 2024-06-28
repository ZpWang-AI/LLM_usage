root_path="/public/home/hongy/zpwang/LLM_Reasoning/logs"
log_path=$(ls -v $root_path | grep -v "/$" | tail -n 1)
cat "${root_path}/${log_path}" | tail -n 3
echo -e "\n"