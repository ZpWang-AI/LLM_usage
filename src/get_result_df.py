import json
import pandas as pd
import numpy as np

from typing import *
from sklearn.metrics import f1_score
from pathlib import Path as path

from IDRR_data import DataFrames
from generate_reasoning import ReasoningArgs, ReasoningGenerator
from process_pred import ReasoningPredProcessor, get_label_id_from_pred
from utils import dump_json


class ReasoningGetResultDfSubtext:
    def __init__(
        self, 
        target_dir_list,
        result_csv_path,
    ) -> None:
        result_df_list = [self.get_result_df_subtext(path(td))for td in target_dir_list if td]
        print([df.shape for df in result_df_list])
        result_df = pd.concat(result_df_list, ignore_index=True)
        result_df = result_df.drop('index', axis=1)
        result_df.to_csv(result_csv_path, index=False)
        
        print(result_df.columns)
        pass
        
    def get_result_df_subtext(self, target_dir:path):
        args = ReasoningArgs.load_json(target_dir/'self.args.json')
        dfs = DataFrames(
            data_name=args.data_name,
            label_level=args.label_level,
            relation=args.relation,
            data_path=args.data_path,
        )
        df = dfs.get_dataframe(split=args.split)
        label_list = dfs.get_label_list()
        
        df_dic = df.to_dict(orient='list')
        if args.n_reasoning_per_sample > 1:
            df_dic = {
                k: [p for p in v for _ in range(args.n_reasoning_per_sample)]
                for k, v in df_dic.items()
            }  # [v1, v2, v3] -> [v1, v1, v2, v2, v3, v3]

        subtext_vals = []
        subtext_results = []
        with open(target_dir/'result.jsonl', 'r', encoding='utf8')as f:
            for line in f.readlines():
                try:
                    cur_dic = json.loads(line)
                    cur_subtext = cur_dic['reasoning'][0]
                    subtext_vals.append(cur_subtext)
                    
                    cur_id = cur_dic['id']
                    cur_pred_id = get_label_id_from_pred(
                        label_list=label_list,
                        pred=cur_dic['reasoning'][1], invalid_pred=len(label_list)
                    )
                    if cur_pred_id == df_dic['conn1sense1id'][cur_id]:
                        subtext_results.append(1)
                    elif any( (not pd.isna(p) and cur_pred_id == p) for p in [
                            df_dic['conn1sense2id'][cur_id],
                            df_dic['conn2sense1id'][cur_id],
                            df_dic['conn2sense2id'][cur_id],
                        ] ):
                        subtext_results.append(2)
                    else:
                        subtext_results.append(0)
                except:
                    import traceback
                    print(traceback.format_exc())
                    exit()
                    pass
        
        dfs.label_level = 'raw'
        result_df = dfs.get_dataframe(split=args.split)
        result_df['subtext'] = subtext_vals
        result_df['subtext_res'] = subtext_results
        print(Counter(subtext_results))
        
        return result_df
        
        # print('Acc:', (len(subtext_results)-subtext_results.count(0))/len(subtext_results))
        
        # del result_dic['index']
        # for key in list(result_dic.keys()):
        #     if 'id' in key:
        #         del result_dic[key]

        # return pd.DataFrame(result_dic, index=None,)
    
    
if __name__ == '__main__':
    ReasoningGetResultDfSubtext(
        target_dir_list='''
        /public/home/hongy/zpwang/LLM_Reasoning/data/groups/subtext_df_pdtb3_l1/gpt3_5.pdtb3.dev_l1.subtext2
        /public/home/hongy/zpwang/LLM_Reasoning/data/groups/subtext_df_pdtb3_l1/gpt3_5.pdtb3.pred_l1.subtext2
        /public/home/hongy/zpwang/LLM_Reasoning/data/groups/subtext_df_pdtb3_l1/gpt3_5.pdtb3.train_l1.subtext2
        '''.split(),
        result_csv_path='/public/home/hongy/zpwang/LLM_Reasoning/data/pdtb3_l1.subtext.csv'
    )