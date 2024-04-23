import pandas as pd
import re

from typing import *


class PromptFiller:
    def __init__(self, df:pd.DataFrame, prompt, ignore=()) -> None:
        self.df = df
        self.prompt = prompt
        self.ignore = ignore
    
    def __iter__(self):
        for p in range(self.df.shape[0]):
            row = self.df.iloc[p]
            res = self.fill_prompt(row, self.prompt, self.ignore)
            yield res            
        # return iter(iter_func())
    
    @classmethod
    def fill_prompt(cls, row:Union[pd.Series, dict], prompt, ignore=()):
        def replace_func(blank:re.Match):
            blank = blank.group()[1:-1]
            if ignore and blank in ignore:
                return ''
            try:
                blank_val = row.__getattr__(blank)
                return str(blank_val) if pd.notna(blank_val) else ''
            except:
                raise Exception(f'row don\'t have blank(attr): {blank}')  
        
        if type(prompt) == str:
            return re.sub(r'\{[^{}]*\}', replace_func, prompt)
        elif type(prompt) == dict:
            return {
                k: cls.fill_prompt(row, v, ignore=ignore)
                for k,v in prompt.items()
            }
        elif type(prompt) == list:
            return [cls.fill_prompt(row, p, ignore=ignore)for p in prompt]
        else:
            raise Exception('wrong type of prompt')
            
    
if __name__ == '__main__':
    from dataframes import DataFrames
    pdtb2_df = DataFrames(
        data_name='pdtb2',
        label_level='level2',
        relation='Implicit',
        data_path=r'/public/home/hongy/zpwang/LLM_Reasoning/data/used/pdtb2.p1.csv',
    )
    df = pdtb2_df.train_df
    df['empty'] = pd.NA
    filler = PromptFiller(
        df, prompt='{conn1sense1}  {relation} >>{empty}<<'
    )
    for p, fp in enumerate(list(filler)):
        print(fp)
        if p == 3:
            break
        