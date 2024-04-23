import os
import sys

from pathlib import Path as path

SRC_PATH = path(__file__).parent.parent
ROOT_PATH = SRC_PATH.parent
sys.path.insert(0, str(SRC_PATH))

from generate_reasoning import ReasoningGenerator, ReasoningArgs


def main():
    prompt = [
        # '''
        # '''.strip(),
        '''
        The first argument:\n\n{arg1}\n\nThe second argument:\n\n{arg2}\n\nWhat\' the relation between arguments? Answer should be one of (Comparison, Contingency, Expansion, Temporal). Just output one answer simply.
        '''.strip(),
    ]
    
    sample_args = ReasoningArgs(
        prompt=prompt,
        llm_name='gpt-3.5-turbo',
        version='gpt3_5.pdtb3.pred_l1.init',  # TODO
        data_name='pdtb3',
        label_level='level1',
        relation='Implicit',
        data_path=ROOT_PATH/'data'/'used'/'pdtb3.p1.csv',
        split='test',
        n_reasoning_per_sample=1,
        max_sample=-1,
    )
    sample_generator = ReasoningGenerator(sample_args)
    sample_generator.get_chat_response_json()
    
    
if __name__ == '__main__':
    main()