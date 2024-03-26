import pandas as pd
import json
import os
import re

from typing import *


LEVEL1_LABEL_LIST = [
    "Comparison",
    "Contingency",
    "Expansion",
    "Temporal"
]
LEVEL2_LABEL_LIST = {
    'pdtb2': [
        "Comparison.Contrast",
        "Comparison.Concession",

        "Contingency.Cause",
        "Contingency.Pragmatic cause",

        "Expansion.Conjunction",
        "Expansion.Restatement",
        "Expansion.Instantiation",
        "Expansion.List",
        "Expansion.Alternative",

        "Temporal.Asynchronous",
        "Temporal.Synchrony"
    ],
    'pdtb3': [
        "Comparison.Concession",
        "Comparison.Contrast",
        
        "Contingency.Cause",
        "Contingency.Purpose",
        "Contingency.Cause+Belief",
        "Contingency.Condition",

        "Expansion.Conjunction",
        "Expansion.Level-of-detail",
        "Expansion.Instantiation",
        "Expansion.Manner",
        "Expansion.Substitution",
        "Expansion.Equivalence",
        
        "Temporal.Asynchronous",
        "Temporal.Synchrony"
    ],
    'conll': [
        'Comparison.Concession',
        'Comparison.Contrast',
        
        'Contingency.Cause.Reason',
        'Contingency.Cause.Result',
        'Contingency.Condition',
        
        'Expansion.Alternative',
        'Expansion.Alternative.Chosen alternative',
        'Expansion.Conjunction',
        'Expansion.Exception',
        'Expansion.Instantiation',
        'Expansion.Restatement',
        
        'Temporal.Asynchronous.Precedence',
        'Temporal.Asynchronous.Succession',
        'Temporal.Synchrony'
    ]
}


class DataFrame:
    def __init__(
        self,
        data_name:Literal['pdtb2', 'pdtb3', 'conll'],
        relation:Literal['Implicit', 'Explicit', 'All']='Implicit',
        label_level:Literal['level1', 'level2']='level1',
        data_path=None,
    ) -> None:
        self.data_name = data_name
        self.label_level = label_level
        self.relation = relation
        
        self.train_df = pd.DataFrame()
        self.dev_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.blind_test_df = pd.DataFrame()
        
        if data_path:
            self.build_dataframe(data_path=data_path)
        
    def build_dataframe(self, data_path):
        df = pd.read_csv(data_path, low_memory=False)
        df = df[df['relation']==self.relation]
        
        for target_df_name in 'train dev test blind-test'.split():
            self.__setattr__(
                f'{target_df_name.replace("-", "_")}_df',
                df[df['split']==target_df_name]
            )
            
    def get_label_list(self):
        if self.data_name not in 'pdtb2 pdtb3 conll'.split():
            raise Exception('wrong data_name')
        if self.label_level == 'level1':
            label_list = LEVEL1_LABEL_LIST
        elif self.label_level == 'level2':
            label_list = LEVEL2_LABEL_LIST[self.data_name]
        else:
            raise Exception('wrong label_level')
        return label_list            
    
    def process_sense(self, sense, return_id=False, label_list=None):
        if pd.isna(sense):
            return ''

        if not label_list:
            label_list = self.get_label_list()
        
        for lid, label in enumerate(label_list):
            if sense.startswith(label):
                return lid if return_id else label
        return -1 if return_id else ''
    
    def process_all_sense(self, return_id=False):
        label_list = self.get_label_list()
        
        def process_df_sense(df):
            for p in range(df.shape[0]):
                row = df.iloc[p]
                for sense_key in 'conn1sense1 conn1sense2 conn2sense1 conn2sense2'.split():
                    row[sense_key] = self.process_sense(
                        row[sense_key],
                        return_id=return_id, 
                        label_list=label_list
                    )
        
        for target_df_name in 'train dev test blind_test':
            target_df_attr = f'{target_df_name}_df'
            if target_df_attr in self.__dict__:
                process_df_sense(self.__getattribute__(target_df_attr))
                
    def label_to_id(self, label):
        label_list = self.get_label_list()
        return label_list.index(label)
    
    def id_to_label(self, lid):
        label_list = self.get_label_list()
        return label_list[lid]
        
    def fill_prompt(self, row:pd.Series, prompt):
        def replace_func(blank:re.Match):
            blank = blank.group()[1:-1]
            if 'sense' not in blank:
                return row.__getattr__(blank)

            blank = blank.split('_')
            if len(blank) == 1:
                return row.__getattr__(blank[0])
            elif len(blank) == 2:
                conn_sense, desc = blank
                if desc == 'level1':
                    self.label_level = desc
                    return self.process_sense(conn_sense, return_id=False)
                elif desc == 'level2':
                    self.label_level = desc
                    return self.process_sense(conn_sense, return_id=False)
                # TODO: add new description of connsense and new processing
                else:
                    raise Exception('wrong description of sense')
                
            raise Exception('wrong blank')
            
        filled_prompt = re.sub(r'\{.*\}', replace_func, prompt)
        return filled_prompt
    
    
if __name__ == '__main__':
    pdtb2_df = DataFrame(
        data_name='pdtb2',
        label_level='level1',
        relation='Implicit',
        data_path=r'D:\0--data\projects\04.00-IDRR_data\used\pdtb2.p1.csv',
    )
    pdtb3_df = DataFrame(
        data_name='pdtb3',
        label_level='level1',
        relation='Implicit',
        data_path=r'D:\0--data\projects\04.00-IDRR_data\used\pdtb3.p1.csv',
    )
    conll_df = DataFrame(
        data_name='conll',
        label_level='level1',
        relation='Implicit',
        data_path=r'D:\0--data\projects\04.00-IDRR_data\used\conll.p1.csv',
    )
    print(pdtb2_df.blind_test_df)
    print(conll_df.blind_test_df)
