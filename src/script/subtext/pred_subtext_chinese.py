import os
import sys

from pathlib import Path as path

SRC_PATH = path(__file__).parent.parent
ROOT_PATH = SRC_PATH.parent
sys.path.insert(0, str(SRC_PATH))

from generate_reasoning import ReasoningGenerator, ReasoningArgs
from process_pred import ReasoningPredProcessor


def main():
    prompt = [
        '''
        论元一：\n{arg1}\n\n论元二：\n{arg2}\n\n这两个论元中的潜台词是什么？
        '''.strip(),
        '''
        论元一和论元二的篇章关系是哪个？\nA. Comparison\nB. Contingency\nC. Expansion\nD. Temporal\n\nAnswer:
        '''.strip(),
    ]
    
    sample_args = ReasoningArgs(
        prompt=prompt,
        llm_name='gpt-3.5-turbo',
        version='gpt3_5.pdtb3.pred_l1.subtext_zh1',  # TODO
        data_name='pdtb3',
        label_level='level1',
        relation='Implicit',
        data_path=ROOT_PATH/'data'/'used'/'pdtb3.p1.csv',
        split='test',
        n_reasoning_per_sample=1,
        max_sample=-1,
    )
    # sample_generator = ReasoningGenerator(sample_args)
    # sample_generator.get_chat_response_json()
    sample_main = ReasoningPredProcessor(sample_args)
    sample_main.get_chat_response_json()
    sample_main.process_pred()
    
    
if __name__ == '__main__':
    main()