from generate_reasoning import *
from chat_api import Messages


class ReasoningGeneratorContinue(ReasoningGenerator):
    def continue_generate(self, new_prompt):
        df = self._dfs.get_dataframe(split=self.split)
        output_dir = self._output_space/self.version
        
        args = load_json(output_dir/'args.json')
        args['new_prompt'] = new_prompt
        dump_json(args, output_dir/'args.new.json', mode='w', indent=4)
        
        result_path = output_dir/'result.jsonl'
        new_result_path = output_dir/'result.new.jsonl'
    
        done_results = load_json(result_path)
        if new_result_path.exists():
            new_results = load_json(new_result_path)
        else:
            new_results = []
        
        progress_bar = tqdm.tqdm(total=len(done_results)-len(new_results))
        for todo_id in range(len(new_results), len(done_results)):
            line = done_results[todo_id]
            data_id = line['data_id']
            done_reasoning = line['reasoning']
            for index, row in df.iterrows():
                if row['data_id'] == data_id:
                    break
            else:
                raise f'no data_id {data_id}'
            
            msg = Messages()
            for usermsg, botmsg in zip(
                PromptFiller.fill_prompt(row, prompt=self.prompt),
                done_reasoning
            ):
                msg.add_user(usermsg)
                msg.add_bot(botmsg)
            msg.add_user(PromptFiller.fill_prompt(row, prompt=new_prompt))
            
            response_list = []
            try:
                for _ in range(self.n_reasoning_per_sample):
                    response = chat_api(messages=dcopy(msg), model=self.llm_name)
                    response_list.append(response)
                    progress_bar.update(1)
            except:
                print(traceback.format_exc())
                exit()
                
            if self.n_reasoning_per_sample == 1:
                response_list = response_list[0]
            
            dump_json(
                target={'data_id': row['data_id'], 'reasoning': response_list}, 
                file_path=new_result_path,
                mode='a',
            )
        progress_bar.close()
        print('All Chatting Tasks are Done')
        
        
    

if __name__ == '__main__':
    prompt = '''
    
Complete the task called Implicit Discourse Relation Recognition (IDRR). Given the pair of arguments and the relation, just output the short and simple reason of the answer directly.\n\nThe first argument:\n\n{arg1}\n\nThe second argument:\n\n{arg2}\n\nRelation:\n\n{conn1sense1}
    
    '''.strip()
    prompt = [
        '''
        The first argument:\n\n{arg1}\n\nThe second argument:\n\n{arg2}\n\nWhat\'s the subtext between the arguments?
        '''.strip(),
        '''
        What\' the relation between arguments? Answer should be one of (Comparison, Contingency, Expansion, Temporal)
        '''.strip(),
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
        max_sample=1,
    )
    sample_generator.start()
    
    
            