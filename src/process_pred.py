from utils_zp import *
from generate_reasoning import ReasoningGenerator

from sklearn.metrics import f1_score, classification_report, confusion_matrix

    

class ReasoningPredProcessor(ReasoningGenerator):
    def process_pred(self, split=None):
        # label_list = self._dfs.label_list
        # num_labels = len(label_list)
        output_dir = self._output_space/self.version
        reasoning_results = load_json(output_dir/'result.jsonl')
        reasoning_dataid_dic = {}
        for line in reasoning_results:
            cur_dataid = line['data_id']
            cur_reasoning = line['reasoning']
            if not isinstance(cur_reasoning, str):
                cur_reasoning = cur_reasoning[-1]+cur_reasoning[-2]
            reasoning_dataid_dic[cur_dataid] = cur_reasoning
        
        reasoning = []
        gt = []
        not_exist = 0
        if split is None:
            split = self.split
        for index, row in self._dfs.get_dataframe(split=split).iterrows():
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
            match_strategy='first exists', 
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
        acc = (confusion_mat*np.eye(len(confusion_mat))).sum() / confusion_mat.sum()
        res = {
            'macro-f1': cls_report['macro avg']['f1-score'],
            'acc': acc,
            'confusion_matrix': confusion_mat.tolist(),
            'cls_report': cls_report,
        }
        print(res)
        dump_json(res, output_dir/'cls_report.json', indent=4)
        
        # f1 = f1_score(gt_vec, pred_vec, average='macro', zero_division=0)
        # f1 *= 100
        # f1_res_path = self.root_path/'f1_score.json'
        # dump_json(f1, f1_res_path)
        # return f1
    

if __name__ == '__main__':
    sample_processor = ReasoningPredProcessor.load_json(
        '/home/user/test/zpwang/LLaMA/exp_space/paper_needed/prompt_llm/gpt-3.5-turbo.pdtb3_top_Implicit_test.base/args.json'
    )
    sample_processor.process_pred(split='test')
    
    
        
        