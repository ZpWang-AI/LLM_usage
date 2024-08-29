#!/bin/bash
# example:
# torun_file="001-test.py"
# `sh run.sh 001`
script_dir="/home/user/test/zpwang/LLM_Reasoning/src/scripts"
log_dir="/home/user/test/zpwang/LLM_Reasoning/logs"
python_exe="/home/user/miniconda3/envs/zpwang_llama/bin/python"

torun_file=$(find $script_dir -type f -name "${1}*")
num_files=$(echo $torun_file | wc -l)
if [ $num_files -ne 1 ]; then
    echo $torun_file
    echo "wrong num of files"
fi

start_time=$(date +%Y-%m-%d-%H-%M-%S)
filename=$(basename "$torun_file")
filename="${filename%.*}"
echo $torun_file
echo "start running"

# source activate zpwang_llama
export MKL_SERVICE_FORCE_INTEL=1
nohup $python_exe $torun_file > "${log_dir}/${start_time}.${filename}.log" 2>&1 &
ps -aux | grep $torun_file | grep -v grep
