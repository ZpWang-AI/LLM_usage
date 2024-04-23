import pandas as pd
import numpy as np
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
    "pdtb2": [
        "Comparison.Concession",
        "Comparison.Contrast",
        
        "Contingency.Cause",
        "Contingency.Pragmatic cause",
        
        "Expansion.Alternative",
        "Expansion.Conjunction",
        "Expansion.Instantiation",
        "Expansion.List",
        "Expansion.Restatement",
        
        "Temporal.Asynchronous",
        "Temporal.Synchrony"
    ],
    "pdtb3": [
        "Comparison.Concession",
        "Comparison.Contrast",
        
        "Contingency.Cause",
        "Contingency.Cause+Belief",
        "Contingency.Condition",
        "Contingency.Purpose",
        
        "Expansion.Conjunction",
        "Expansion.Equivalence",
        "Expansion.Instantiation",
        "Expansion.Level-of-detail",
        "Expansion.Manner",
        "Expansion.Substitution",
        
        "Temporal.Asynchronous",
        "Temporal.Synchronous"
    ],
    "conll": [
        "Comparison.Concession",
        "Comparison.Contrast",
        
        "Contingency.Cause.Reason",
        "Contingency.Cause.Result",
        "Contingency.Condition",
        
        "Expansion.Alternative",
        "Expansion.Alternative.Chosen alternative",
        "Expansion.Conjunction",
        "Expansion.Exception",
        "Expansion.Instantiation",
        "Expansion.Restatement",
        
        "Temporal.Asynchronous.Precedence",
        "Temporal.Asynchronous.Succession",
        "Temporal.Synchrony"
    ]
}


class DataFrames:
    def __init__(
        self,
        data_name:Literal['pdtb2', 'pdtb3', 'conll', None]=None,
        label_level:Literal['level1', 'level2', 'raw']='raw',
        relation:Literal['Implicit', 'Explicit', 'All']='Implicit',
        data_path:str=None,
        # label_use_id=False
    ) -> None:
        self.data_name = data_name
        self.relation = relation 
        self.label_level = label_level
        self.data_path = data_path
        # self.label_use_id = label_use_id
    
        self.df = pd.DataFrame()
        if data_path:
            self.build_dataframe(data_path=data_path)
    
    def build_dataframe(self, data_path):
        self.df = pd.read_csv(data_path, low_memory=False)
    
    def get_dataframe(self, split) -> pd.DataFrame:
        df = self.df[self.df['split']==split]
        if self.relation != 'All':
            df = df[df['relation']==self.relation]
        if self.data_name and self.label_level != 'raw':
            df = self.process_df_sense(df)
        df = df[pd.notna(df['conn1sense1'])]
        df.reset_index(inplace=True)
        return df
    
    @property
    def train_df(self) -> pd.DataFrame:
        return self.get_dataframe('train')
    
    @property
    def dev_df(self) -> pd.DataFrame:
        return self.get_dataframe('dev')
        
    @property
    def test_df(self) -> pd.DataFrame:
        return self.get_dataframe('test')
        
    @property
    def blind_test_df(self) -> pd.DataFrame:
        return self.get_dataframe('blind-test')
            
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
    
    def process_sense(
        self, sense:str,
        label_list=None, 
        irrelevent_sense=pd.NA,
    ) -> Tuple[str, int]:
        if pd.isna(sense):
            return irrelevent_sense, irrelevent_sense

        if not label_list:
            label_list = self.get_label_list()
        
        if sense in label_list:
            return sense, label_list.index(sense)
        for lid, label in enumerate(label_list):
            if sense.startswith(label+'.'):
                return label, lid

        return irrelevent_sense, irrelevent_sense
    
    def process_df_sense(self, df:pd.DataFrame):
        label_list = self.get_label_list()
        
        for sense_key in 'conn1sense1 conn1sense2 conn2sense1 conn2sense2'.split():
            label_values, lid_values = [], []
            for sense in df[sense_key]:
                label, lid = self.process_sense(
                    sense=sense, label_list=label_list, irrelevent_sense=pd.NA,
                )
                label_values.append(label)
                lid_values.append(lid)
            df[sense_key] = label_values
            df[sense_key+'id'] = lid_values
        return df
                
    def label_to_id(self, label):
        label_list = self.get_label_list()
        return label_list.index(label)
    
    def id_to_label(self, lid):
        label_list = self.get_label_list()
        return label_list[lid]

    
if __name__ == '__main__':
    from pathlib import Path as path
    data_root_path = path('/public/home/hongy/zpwang/IDRR_ConnT5/data')
    pdtb2_df = DataFrames(
        data_name='pdtb2',
        label_level='level2',
        relation='Implicit',
        data_path=data_root_path/'used'/'pdtb2.p1.csv',
    )
    pdtb3_df = DataFrames(
        data_name='pdtb3',
        label_level='level2',
        relation='Implicit',
        data_path=data_root_path/'used'/'pdtb3.p1.csv',
    )
    conll_df = DataFrames(
        data_name='conll',
        label_level='level2',
        relation='Implicit',
        data_path=data_root_path/'used'/'conll.p1.csv',
    )
    # print(pdtb2_df.blind_test_df)
    # print('='*30)
    # print(conll_df.blind_test_df)
    # print('='*30)

    sample_df = conll_df
    sample_df.label_level = 'level2'
    # sample_df.label_use_id = True
    # sample_df.label_level = 'level2'
    # sample_df.label_use_id = True
    sample_train_df = sample_df.train_df
    print(sample_train_df.conn1sense2)
    # print(sample_train_df.index)
    # for p in sorted(set(sample_train_df.conn1sense1)):
    #     print(p)
    # if sample_df.label_level != 'raw':
    #     print(len(set(sample_train_df.conn1sense1))==len(sample_df.get_label_list()))
    # print(set(sample_train_df.relation))
    