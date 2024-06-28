from utils_zp.common_import import *

import pandas as pd

from chat_api import chat_api
from IDRR_data import PromptFiller, IDRRDataFrames
from utils_zp import ExpArgs, dump_json, load_json


class ReasoningGenerator(ExpArgs):
    def __init__(
        self,
        prompt,
        llm_name:str,    
        desc:str,
        output_space,
        
        dfs:IDRRDataFrames,
        split:Literal['train', 'dev', 'test', 'all'],
        
        n_reasoning_per_sample=1,
        max_sample=-1,
        
        *args, **kwargs
    ) -> None:
        self.prompt = prompt
        self.llm_name = llm_name
        self.desc = desc
        self._output_space = output_space
        
        self._dfs = dfs
        self.IDRR_dataframes = dfs.json
        self.split = split
        
        self.n_reasoning_per_sample = n_reasoning_per_sample
        if max_sample < 0:
            max_sample = 10**20
        self.max_sample = max_sample
        self.set_create_time()
        
        self._version_info_list = [
            self.llm_name,
            f'{dfs}_{split}',
            self.desc, 
        ]
        
    @classmethod
    def load_json(cls, json_path, overwrite_existing=True):
        # res = super().load_json(json_path)
        args = load_json(json_path)
        dfs = IDRRDataFrames(
            **args['IDRR_dataframes']
        )
        args['dfs'] = dfs
        args['output_space'] = path(json_path).parent
        return cls(**args)
        # res = cls(**load_json(json_path))
        # res._dfs = 
        # return res
    
    def start(self):
        df = self._dfs.get_dataframe(split=self.split)
        
        output_path = path(self._output_space)/self.version
        if output_path.exists():
            if input('output_path exist, continue? y/n\n') != 'y':
                exit()
        else:
            make_path(dir_path=output_path)
        dump_json(self.json, output_path/'args.json', mode='w', indent=4)
        result_path = output_path/'result.jsonl'
    
        dealt_data_id_set = set()
        if result_path.exists():
            for line in load_json(result_path):
                dealt_data_id_set.add(line['data_id'])
        
        progress_bar = tqdm.tqdm(total=min(self.max_sample, len(df)-len(dealt_data_id_set))*self.n_reasoning_per_sample)
        for index, row in df.iterrows():
            if row['data_id'] in dealt_data_id_set:
                continue
            
            query = PromptFiller.fill_prompt(row, self.prompt)
            response_list = []
            try:
                for _ in range(self.n_reasoning_per_sample):
                    response = chat_api(content=query, model=self.llm_name)
                    response_list.append(response)
                    progress_bar.update(1)
            except:
                print(traceback.format_exc())
                exit()
                
            if self.n_reasoning_per_sample == 1:
                response_list = response_list[0]
            
            dump_json(
                target={'data_id': row['data_id'], 'reasoning': response_list}, 
                file_path=result_path,
                mode='a',
            )
        progress_bar.close()
        print('All Chatting Tasks are Done')
        
    def get_result_df(self):
        raise Exception('TODO')
        result_dic = self.df.to_dict(orient='list')
        if self.n_reasoning_per_sample > 1:
            result_dic = {
                k: [p for p in v for _ in range(self.n_reasoning_per_sample)]
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
    
    dfs = IDRRDataFrames(
        data_name='pdtb3', data_level='top', data_relation='Implicit',
        data_path='/home/qwe/test/zpwang/IDRR_data/data/used/pdtb3_top_implicit.subtext2.csv'
    )
    sample_generator = ReasoningGenerator(
        prompt=prompt,
        llm_name='gpt-4-turbo',
        desc=f'subtext',
        output_space='/home/qwe/test/zpwang/LLM_Reasoning/data/reasoning',
        dfs=dfs,
        split='test',
        n_reasoning_per_sample=1,
        max_sample=1,
    )
    sample_generator.start()
    
    
            