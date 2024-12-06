import os
import sys

from pathlib import Path as path

SRC_PATH = path(__file__).parent.parent
ROOT_PATH = SRC_PATH.parent
sys.path.insert(0, str(SRC_PATH))

from generate_reasoning import ReasoningGenerator, ReasoningArgs
from process_pred import ReasoningPredProcessor


def main():
    prompt = '''
The task is to determine whether they have a temporal, comparative, contingency, or extensional relationship.

Example 1-
Input: Text fragment1: "there's a satisfaction in going against the rules" Text fragment2: "he means the rule that a player can't cut it after a certain age"
Output: Extension
Example 2-
Input: Text fragment1: "that prevents the production of pollen" Text fragment2: "the gene can prevent a plant from fertilizing itself"
Output: Contingency
Example 3-
Input: Text fragment1: "he was heralded by a trumpet fanfare" Text fragment2: "the judge marched down the center aisle in his flowing black robe"
Output: Temporal
Example 4-
Input: Text fragment1: "however, the maximum coupon at which the notes can be reset is 16 1/4%" Text fragment2: "the minimum coupon is 13 3/4%"
Output: Comparison

Input: Text fragment1:"{arg1}" Text fragment2:"{arg2}"
Output: <label>
'''.strip()
    # print(prompt)
    # return

    prompt = [prompt]
    
    sample_args = ReasoningArgs(
        version='gpt3_5.pdtb3.pred_l1.base3_IICOT1',  # !!! TODO !!!
        prompt=prompt,
        llm_name='gpt-3.5-turbo',
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