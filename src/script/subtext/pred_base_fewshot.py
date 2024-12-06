import os
import sys

from pathlib import Path as path

SRC_PATH = path(__file__).parent.parent
ROOT_PATH = SRC_PATH.parent
sys.path.insert(0, str(SRC_PATH))

from generate_reasoning import ReasoningGenerator, ReasoningArgs
from process_pred import ReasoningPredProcessor


def main():
    sample1 = 'Argument 1:\n'
    sample2 = ''
    sample3 = ''
    
    prompt_task = '''

Argument 1:
{arg1}

Argument 2:
{arg2}

Question: What is the discourse relation between Argument 1 and Argument 2?
A. Comparison
B. Contingency
C. Expansion
D. Temporal

Example 1:
\tArgument 1:
\thowever, the maximum coupon at which the notes can be reset is 16 1/4%
\t
\tArgument 2:
\tthe minimum coupon is 13 3/4%
\t  
\tAnswer:
\tA. Comparison
    '''.strip()
# Example 2:
# \tArgument 1:
# \tthat prevents the production of pollen
# \t
# \tArgument 2:
# \tthe gene can prevent a plant from fertilizing itself
# \t
# \tAnswer:
# \tB. Contingency
    
# Example 3:
# \tArgument 1: 
# \tthere's a satisfaction in going against the rules
# \t
# \tArgument 2: 
# \the means the rule that a player can't cut it after a certain age
# \t
# \tAnswer: 
# \tC. Expansion
    
# Example 4:
# \tArgument 1:
# \the was heralded by a trumpet fanfare
# \t
# \tArgument 2:
# \tthe judge marched down the center aisle in his flowing black robe
# \t
# \tAnswer:
# \tD. Temporal
    
    prompt = [f'{prompt_task}']
    
    
    sample_args = ReasoningArgs(
        prompt=prompt,
        llm_name='gpt-3.5-turbo',
        version='gpt3_5.pdtb3.pred_l1.base_fewshot1_1',  # TODO
        data_name='pdtb3',
        label_level='level1',
        relation='Implicit',
        data_path=ROOT_PATH/'data'/'used'/'pdtb3.p1.csv',
        split='test',
        n_reasoning_per_sample=1,
        max_sample=-1,
    )
    sample_main = ReasoningPredProcessor(sample_args)
    sample_main.get_chat_response_json()
    sample_main.process_pred()
    
if __name__ == '__main__':
    main()
    