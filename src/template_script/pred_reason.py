from utils_zp.common_import import *

add_sys_path(__file__, 2)

SRC_PATH = path(__file__).parent.parent
ROOT_PATH = SRC_PATH.parent

from IDRR_data import IDRRDataFrames
from generate_reasoning import ReasoningGenerator
# from process_pred import ReasoningPredProcessor


def main():
    prompt = [
        '''
Argument 1:
{arg1}

Argument 2:
{arg2}

What's the implicit meaning between the arguments?
        '''.strip(),
        '''
What is the discourse relation between Argument 1 and Argument 2?
A. Comparison
B. Contingency
C. Expansion
D. Temporal

Answer:
        '''.strip(),
        '''Just output the answer directly.'''
    ]
    
    dfs = IDRRDataFrames(
        data_name='pdtb3', data_level='top', data_relation='Implicit',
        data_path='/home/user/test/zpwang/IDRR_data/data/used/pdtb3_top_implicit.subtext2.csv'
    )
    sample_generator = ReasoningGenerator(
        prompt=prompt,
        llm_name='gpt-4-turbo',
        desc=f'subtext',
        output_space='/home/user/test/zpwang/LLM_Reasoning/data/reasoning',
        dfs=dfs,
        split='test',
        n_reasoning_per_sample=1,
        max_sample=-1,
    )
    sample_generator.start()
    
    
if __name__ == '__main__':
    main()