import os
import json
import pandas as pd

from typing import *
from tqdm import tqdm
from pathlib import Path as path

from chat_api import chat_api
# from dataframe import DataFrame
from IDRR_data import DataFrames, PromptFiller
# from reason_args import ReasonArgs
from utils import AttrDict


class ReasoningArgs(AttrDict):
    def __init__(
        self,
        prompt,
        llm_name:str,    
        version:str,
        
        data_name:Literal['pdtb2', 'pdtb3', 'conll'],
        label_level:Literal['level1', 'level2', 'raw'],
        relation:Literal['Implicit', 'Explicit', 'All'],
        data_path:str,
        split:Literal['train', 'dev', 'test', 'blind_test'],
        
        n_reasoning_per_sample=1,
        max_sample=-1,
        create_time=None,
    ) -> None:
        self.prompt = prompt
        self.llm_name = llm_name
        self.version = version
        
        self.data_name = data_name
        self.label_level = label_level
        self.relation = relation
        self.data_path = data_path
        self.split = split
        
        self.n_reasoning_per_sample = n_reasoning_per_sample
        if max_sample < 0:
            max_sample = 10**20
        self.max_sample = max_sample
        self.set_create_time(create_time=create_time)
            
    # def dump_json(self, arg_space, overwrite=False):
    #     json_path = path(arg_space)/f'{self.create_time}.{self.llm_name}.{self.version}.json'
    #     self._dump_json(json_path, overwrite=overwrite)
    

class ReasoningGenerator:
    def __init__(
        self, 
        reasoning_args:ReasoningArgs
    ) -> None:
        self.args = reasoning_args
        
        self.dfs = DataFrames(
            data_name=self.args.data_name,
            label_level=self.args.label_level,
            relation=self.args.relation,
            data_path=self.args.data_path,
        )
        self.df = self.dfs.get_dataframe(split=self.args.split)

        self.root_path = path(__file__).parent.parent/'data'/'reasoning'/self.args.version
        self.args._dump_json(self.root_path/'self.args.json', overwrite=False)
        self.result_path = self.root_path/'result.json'
    
    def get_chat_response_json(self):
        if path(self.result_path).exists():
            with open(self.result_path, 'r', encoding='utf8')as f:
                start_id = len(f.readlines())
        else:
            start_id = 0
        end_id = min(self.df.shape[0], start_id+self.args.max_sample)
        
        progress_bar = tqdm(total=(end_id-start_id)*self.args.n_reasoning_per_sample)
        for pid in range(start_id, end_id):
            
            row = self.df.iloc[pid]
            query = PromptFiller.fill_prompt(row, prompt)
            response_list = []
            try:
                for _ in range(self.args.n_reasoning_per_sample):
                    response = chat_api(content=query, model=self.args.llm_name)
                    response_list.append(response)
                    progress_bar.update(1)
            except:
                import traceback
                print(traceback.format_exc())
                exit()
                
            if self.args.n_reasoning_per_sample == 1:
                response_list = response_list[0]
            
            with open(self.result_path, 'a', encoding='utf8')as f:
                json.dump({'id': pid, 'reasoning': response_list}, f)
                f.write('\n')
        progress_bar.close()
        print('All Chatting Tasks are Done')
        
    def get_result_df(self):
        raise Exception('TODO')
        result_dic = self.df.to_dict(orient='list')
        if self.args.n_reasoning_per_sample > 1:
            result_dic = {
                k: [p for p in v for _ in range(self.args.n_reasoning_per_sample)]
                for k, v in result_dic.items()
            }  # [v1, v2, v3] -> [v1, v1, v2, v2, v3, v3]

        reasoning_vals = []
        with open(self.result_path, 'r', encoding='utf8')as f:
            for line in f.readlines():
                try:
                    cur_val = json.loads(line)['reasoning']
                    reasoning_vals.extend(cur_val)
                except:
                    pass
        result_dic['reasoning'] = reasoning_vals
        result_df = pd.DataFrame(result_dic)
        result_df_list = [result_df, self.dfs.dev_df, self.dfs.test_df]
        if self.dfs.blind_test_df.shape[0]:
            result_df_list.append(self.dfs.blind_test_df)
        result_df = pd.concat(result_df_list, ignore_index=True)
        result_csv = '.'.join(self.result_path.split('.')[:-1]+['csv'])
        result_df.to_csv(result_csv, index=False)
        
        result_df = pd.read_csv(result_csv)
        print()
        print(result_df.shape)
        print(result_df.columns)
        pass
    

if __name__ == '__main__':
    prompt = '''
    
Complete the task called Implicit Discourse Relation Recognition (IDRR). Given the pair of arguments and the relation, just output the short and simple reason of the answer directly.\n\nThe first argument:\n\n{arg1}\n\nThe second argument:\n\n{arg2}\n\nRelation:\n\n{conn1sense1}
    
    '''.strip()
    prompt = [
        '''
        The first argument:\n\n{arg1}\n\nThe second argument:\n\n{arg2}\n\nWhat\'s the subtext between the arguments?
        '''.strip(),
        '''
        What\' the relation between arguments? Answer should be one of (Comparison, Contingency, Expansion, Temporal)
        '''.strip(),
    ]
    
    sample_args = ReasoningArgs(
        prompt=prompt,
        llm_name='gpt-3.5-turbo',
        version='gpt3_5.pdtb3.pred_l1.subtext',
        data_name='pdtb3',
        label_level='raw',
        relation='Implicit',
        data_path='/public/home/hongy/zpwang/LLM_Reasoning/data/used/pdtb3.p1.csv',
        split='test',
        n_reasoning_per_sample=1,
        max_sample=-1,
    )
    sample_generator = ReasoningGenerator(sample_args)
    sample_generator.get_chat_response_json()
    
    
            