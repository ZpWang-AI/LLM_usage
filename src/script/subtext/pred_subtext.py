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
        Argument 1:\n{arg1}\n\nArgument 2:\n{arg2}\n\nWhat\'s the implicit meaning between the arguments?
        '''.strip(),
        '''
        What is the discourse relation between Argument 1 and Argument 2?\nA. Comparison\nB. Contingency\nC. Expansion\nD. Temporal\n\nAnswer:
        '''.strip(),
    ]
    
    sample_args = ReasoningArgs(
        prompt=prompt,
        llm_name='gpt-3.5-turbo',
        version='gpt3_5.pdtb2.pred_l1.subtext2',  # TODO
        data_name='pdtb2',
        label_level='level1',
        relation='Implicit',
        data_path=ROOT_PATH/'data'/'used'/'pdtb2.p1.csv',
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