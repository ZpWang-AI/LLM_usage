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
        '''Please just output the answer directly.'''
    ]
    
    dfs = IDRRDataFrames(
        data_name='pdtb3', data_level='top', data_relation='Implicit',
        data_path='/home/user/test/zpwang/IDRR_data/data/used/pdtb3_top_implicit.csv'
    )
    # >>> start new reasoning
    sample_generator = ReasoningGenerator(
        prompt=prompt,
        llm_name='gpt-3.5-turbo',
        desc=f'subtext',
        output_space='/home/user/test/zpwang/LLM_Reasoning/data/reasoning',
        dfs=dfs,
        split='test',
        n_reasoning_per_sample=1,
        max_sample=-1,
    )
    
    # >>> continue old reasoning
    # sample_generator = ReasoningGenerator.load_json(
    #     json_path='/home/user/test/zpwang/LLM_Reasoning/data/reasoning/gpt-3.5-turbo.pdtb3_top_Implicit_test.subtext/args.json',
    # )
    # sample_generator.max_sample = 10**9
    
    sample_generator.start()
    
    
if __name__ == '__main__':
    main()