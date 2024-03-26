import sys
sys.path.insert(0, '/public/home/hongy/zpwang/LLM_API')
sys.path.insert(0, '/public/home/hongy/zpwang/LLM_API/llm_reasoning')

import os
import json
import pandas as pd

from pathlib import Path as path

from chat import chat, model_choices
from dataframe import DataFrame
from reason_frame import ReasonFrame
from tqdm import tqdm


class Generator:
    def __init__(
        self, 
        prompt, 
        data_name,
        label_level,
        relation,
        data_path,
        model:model_choices,
        version='base',
        
        n_reason_per_sample=1,
        
        max_sample=-1,
    ) -> None:
        if max_sample < 0:
            max_sample = 10**20
        
        reason_frame = ReasonFrame(
            prompt=prompt,
            version=version,
            data_name=data_name,
            label_level=label_level,
            relation=relation,
            model=model_choices,
        )
        reason_frame.dump_json(
            prompt_space='/public/home/hongy/zpwang/LLM_API/llm_reasoning/data/reason/prompt'
        )
        
        df = DataFrame(
            data_name=data_name,
            label_level=label_level,
            relation=relation,
            data_path=data_path,
        )
        train_df = df.train_df
        
        result_name = f'{data_name}_{model}_{version}'
        result_path = f'/public/home/hongy/zpwang/LLM_API/llm_reasoning/data/reason/{result_name}.jsonl'
        
        if path(result_path).exists():
            with open(result_path, 'r', encoding='utf8')as f:
                start_id = len(f.readlines())
        else:
            start_id = 0
            
        # file_mark = FileMark()
        
        end_id = min(train_df.shape[0], start_id+max_sample)
        for pid in tqdm(range(start_id, end_id)):
            # file_mark.write_mark(f'run_{pid}vs{end_id}')
            
            row = train_df.iloc[pid]
            fp = df.fill_prompt(row, prompt)
            try:
                ans = chat(content=fp, model=model)
            except:
                break
            with open(result_path, 'a', encoding='utf8')as f:
                json.dump({'id': pid, 'reason': ans}, f)
                f.write('\n')
            
            max_sample -= 1
            if max_sample == 0:
                break
        
        # file_mark.write_mark('stop')
    

if __name__ == '__main__':
    Generator(
        prompt_path='/public/home/hongy/zpwang/LLM_API/llm_reasoning/prompt/base.txt',
        data_name='pdtb3',
        data_path='/public/home/hongy/zpwang/LLM_API/llm_reasoning/data/used/pdtb3.p1.csv',
        model='gpt-3.5-turbo',
        version='v1',
        max_sample=-1,
    )
            