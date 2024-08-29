from utils_zp.common_import import *

import pandas as pd

from chat_api import chat_api, chat_api_template, API_PLACEHOLDER
from IDRR_data import PromptFiller, IDRRDataFrames
from utils_zp import ExpArgs, dump_json, load_json


class ReasoningGenerator(ExpArgs):
    def __init__(
        self,
        template,
        llm_name:str,    
        desc:str,
        output_space,
        
        dfs:IDRRDataFrames,
        split:Literal['train', 'dev', 'test', 'all'],
        
        max_sample=-1,
        
        *args, **kwargs
    ) -> None:
        self.template = template
        self.llm_name = llm_name
        self.desc = desc
        self._output_space = output_space
        
        self._dfs = dfs
        self.IDRR_dataframes = dfs.json
        self.split = split
        
        if max_sample < 0:
            max_sample = 10**20
        self.max_sample = max_sample
        self.set_create_time()
        
        self._version_info_list = [
            self.llm_name,
            f'{dfs}_S{split}',
            self.desc, 
        ]
        
        self.merge_dict(kwargs)
        
    @classmethod
    def load_json(cls, json_path):
        args = load_json(json_path)
        dfs = IDRRDataFrames(
            **args['IDRR_dataframes']
        )
        args['dfs'] = dfs
        args['output_space'] = path(json_path).parent.parent
        sample = cls(**args)
        return sample
    
    def start(self):
        df = self._dfs.get_dataframe(split=self.split)
        
        output_dir = path(self._output_space)/self.version
        if not output_dir.exists():
            make_path(dir_path=output_dir)
        args_path = output_dir/'args.json' # create new args or load old args
        if not args_path.exists():
            dump_json(self.json, args_path, mode='w', indent=4)
        else:
            args_info = load_json(args_path)
            assert self.create_time == args_info['create_time']
        result_path = output_dir/'result.jsonl'
    
        dealt_data_id_set = set()
        if result_path.exists():
            for line in load_json(result_path):
                dealt_data_id_set.add(line['data_id'])
        
        progress_bar = tqdm.tqdm(total=min(self.max_sample, len(df)))  # done_reasoning / total_reasoning
        for index, row in df.iterrows():
            if row['data_id'] in dealt_data_id_set:
                progress_bar.update(1)
                progress_bar.display()
                continue
            if len(dealt_data_id_set) >= self.max_sample:
                break
            dealt_data_id_set.add(row['data_id'])
            
            template = PromptFiller.fill_prompt(row, self.template)
            response = chat_api_template(template=template, model=self.llm_name)
            progress_bar.update(1)
            
            dump_json(
                target={'data_id': row['data_id'], 'reasoning': response}, 
                file_path=result_path,
                mode='a',
            )
        progress_bar.close()
        print('All Chatting Tasks are Done')
    

if __name__ == '__main__':
#     prompt = '''
    
# Complete the task called Implicit Discourse Relation Recognition (IDRR). Given the pair of arguments and the relation, just output the short and simple reason of the answer directly.\n\nThe first argument:\n\n{arg1}\n\nThe second argument:\n\n{arg2}\n\nRelation:\n\n{conn1sense1}
    
#     '''.strip()
    template = [
        '''
        The first argument:\n\n{arg1}\n\nThe second argument:\n\n{arg2}\n\nWhat\'s the subtext between the arguments?
        '''.strip(),
        API_PLACEHOLDER,
        '''
        What\' the relation between arguments? Answer should be one of (Comparison, Contingency, Expansion, Temporal)
        '''.strip(),
        API_PLACEHOLDER,
    ]
    
    dfs = IDRRDataFrames(
        data_name='pdtb3', data_level='top', data_relation='Implicit',
        data_path='/home/user/test/zpwang/IDRR_data/data/used/pdtb3_top_implicit.subtext2.csv'
    )
    sample_generator = ReasoningGenerator(
        template=template,
        llm_name='gpt-3.5-turbo',
        desc=f'subtext',
        # output_space='/home/user/test/zpwang/LLM_Reasoning/data/reasoning',
        output_space='/home/user/test/zpwang/LLM_Reasoning/data/',
        dfs=dfs,
        split='test',
        max_sample=1,
    )
    sample_generator.start()
    
    
            