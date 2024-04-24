import json
import pandas as pd
import numpy as np

from typing import *
from sklearn.metrics import f1_score

from generate_reasoning import ReasoningArgs, ReasoningGenerator
from utils import dump_json


def get_label_id_from_pred(label_list:List[str], pred:str, invalid_pred:int=-1):
    res = invalid_pred
    res_pid = 10**20

    pred = pred.lower()
    for lid, label in enumerate(label_list):
        label = label.lower()
        if label in pred:
            pid = pred.index(label)
            if pid < res_pid:
                res = lid
                res_pid = pid
    
    return res
    

class ReasoningPredProcessor(ReasoningGenerator):
    # def 
    def process_pred(self):
        label_list = self.dfs.get_label_list()
        num_labels = len(label_list)

        with open(self.result_path, 'r', encoding='utf8')as f:
            preds = []
            for line in f.readlines():
                line = line.strip()
                if line:
                    line = json.loads(line)['reasoning']
                    if isinstance(line, str):
                        preds.append(line)
                    elif hasattr(line, '__iter__'):
                        preds.append(list(line)[-1])
                    else:
                        print(line)
                        raise Exception('wrong pred')
        pred_ids = [
            get_label_id_from_pred(
                label_list=label_list, pred=p, invalid_pred=num_labels,
            ) for p in preds
        ]
        pred_vec = np.eye(num_labels+1, num_labels)[pred_ids]
        
        def get_gt_vec(gt:pd.Series):
            gt = [num_labels if pd.isna(p) else int(p) for p in gt]
            return np.eye(num_labels+1, num_labels, dtype=int)[gt]
        
        gt_vec = np.zeros((self.df.shape[0], num_labels), dtype=int)
        for id_key in 'conn1sense1 conn1sense2 conn2sense1 conn2sense2'.split():
            id_key += 'id'
            gt_vec += get_gt_vec(self.df[id_key])
        
        pred_vec = pred_vec!=0
        gt_vec = gt_vec!=0
        f1 = f1_score(gt_vec, pred_vec, average='macro', zero_division=0)
        f1 *= 100
        f1_res_path = self.root_path/'f1_score.json'
        dump_json(f1, f1_res_path)
        return f1
    

if __name__ == '__main__':
    sample_args = ReasoningArgs.load_json(
        '/public/home/hongy/zpwang/LLM_Reasoning/data/reasoning/gpt3_5.pdtb3.pred_l1.base2/self.args.json'
        # '/public/home/hongy/zpwang/LLM_Reasoning/data/reasoning/gpt3_5.pdtb3.pred_l1.init/self.args.json'
    )
    sample_processor = ReasoningPredProcessor(sample_args)
    print(sample_processor.process_pred())
        
        