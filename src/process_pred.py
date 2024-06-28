from utils_zp.common_import import *

from sklearn.metrics import f1_score, classification_report, confusion_matrix

from generate_reasoning import ReasoningGenerator
from utils_zp import dump_json, load_json, postprocess_generation_res_to_lid
    

class ReasoningPredProcessor(ReasoningGenerator):
    def process_pred(self):
        # label_list = self._dfs.label_list
        # num_labels = len(label_list)

        reasoning_results = load_json(self._output_space/'result.jsonl')
        reasoning_dataid_dic = {}
        for line in reasoning_results:
            cur_dataid = line['data_id']
            cur_reasoning = line['reasoning']
            if not isinstance(cur_reasoning, str):
                cur_reasoning = cur_reasoning[-1]
            reasoning_dataid_dic[cur_dataid] = cur_reasoning
        
        reasoning = []
        gt = []
        not_exist = 0
        for index, row in self._dfs.get_dataframe(split=self.split).iterrows():
            cur_dataid = row['data_id']
            if cur_dataid in reasoning_dataid_dic:
                reasoning.append(reasoning_dataid_dic[cur_dataid])
                gt.append(row['label11'])
            else:
                not_exist += 1
        print(f'{not_exist} samples do not exist')
        # exit()
        
        res_to_lid = postprocess_generation_res_to_lid(
            pred=reasoning, gt=gt, 
            match_strategy='last exists', 
            # lower_results=True
        )
        reasoning = res_to_lid['pred']
        gt = res_to_lid['gt']
        label_list = res_to_lid['label_list']
        
        confusion_mat = confusion_matrix(
            y_true=gt, y_pred=reasoning, labels=list(range(len(label_list))),
        )
        cls_report = classification_report(
            y_true=gt, y_pred=reasoning, labels=list(range(len(label_list))),
            target_names=label_list, output_dict=True
        )
        print(confusion_mat)
        print_sep()
        print(classification_report(
            y_true=gt, y_pred=reasoning, labels=list(range(len(label_list))),
            target_names=label_list, output_dict=False
        ))
        # print(cls_report)
        # return
        res = {
            'macro-f1': cls_report['macro avg']['f1-score'],
            'confusion_matrix': confusion_mat.tolist(),
            'cls_report': cls_report,
        }
        dump_json(res, self._output_space/'cls_report.json', indent=4)
        # f1 = f1_score(gt_vec, pred_vec, average='macro', zero_division=0)
        # f1 *= 100
        # f1_res_path = self.root_path/'f1_score.json'
        # dump_json(f1, f1_res_path)
        # return f1
    

if __name__ == '__main__':
    sample_processor = ReasoningPredProcessor.load_json(
        '/home/qwe/test/zpwang/LLM_Reasoning/data/reasoning/gpt-4-turbo.pdtb3_top_Implicit_test.base/args.json'
    )
    sample_processor.process_pred()
    # sample_args = ReasoningArgs.load_json(
    #     '/public/home/hongy/zpwang/LLM_Reasoning/data/reasoning/gpt3_5.pdtb3.pred_l1.base2/self.args.json'
    #     # '/public/home/hongy/zpwang/LLM_Reasoning/data/reasoning/gpt3_5.pdtb3.pred_l1.init/self.args.json'
    # )
    # sample_processor = ReasoningPredProcessor(sample_args)
    # print(sample_processor.process_pred())
        
        