import re
import datetime
import json
import openai
import time
import traceback

from typing import *
from pathlib import Path as path
from copy import deepcopy as dcopy
from functools import partial
from openai import OpenAI

# === API usage record path ===
RECORD_DIR_PATH = path(__file__).parent.parent/'api_record'
RECORD_PATH = RECORD_DIR_PATH/'record.jsonl'
path(RECORD_DIR_PATH).mkdir(parents=True, exist_ok=True)

# === API KEY ===
from api_key_file import API_KEY

usage_bill_dic = {
    'gpt-3.5-turbo': {'prompt_tokens': 1.5e-6, 'completion_tokens': 4.5e-6,},
    'claude-3-opus-20240229': {'prompt_tokens': 15e-6, 'completion_tokens': 75e-6,},
    'gpt-4-turbo': {'prompt_tokens': 10e-6, 'completion_tokens': 30e-6,},
}

_API_KEY_DIC = {}
for kv in re.split(r'\n{2,}', API_KEY.strip()):
    kv = kv.split('\n')
    _API_KEY_DIC[kv[0]] = kv[1]

QUERY_PLACEHOLDER = '<API>'


class Messages: 
    def __init__(self, messages:Union['Messages', str, List[dict], None]=None) -> None:
        if isinstance(messages, type(self)):
            self.value = messages.value
        elif isinstance(messages, str):
            self.value = []
            self.add_user(messages)
        elif isinstance(messages, (list, tuple)):
            if all(isinstance(p, dict)for p in messages):
                self.value = messages
        elif messages is None:
            self.value = []
        
        if not hasattr(self, 'value'):
            print(messages)
            print(type(messages))
            raise Exception('wrong type of messages')

    def __add_content(self, role, content):
        self.value.append({'role': role, 'content': content})
        
    def add_user(self, content):
        self.__add_content('user', content)
        
    def add_bot(self, content):
        self.__add_content('assistant', content)

    def add_system(self, content):
        self.__add_content('system', content)
        
    def __repr__(self) -> str:
        res = []
        for piece in self.value:
            res.append(f'=== {piece["role"]} ===')
            res.append(piece['content']+'\n')
        res.append('='*20)
        return '\n'.join(res)

    def __len__(self):
        return sum(p['role']=='assistant' for p in self.value)
    

class APIRecordManager:
    @classmethod
    def dump_record(cls, model, usage, messages:Messages, record_path=RECORD_PATH):
        usage_dic = {}
        for k,v in dict(usage).items():
            try:
                usage_dic[k] = int(v)
            except:
                usage_dic[k] = str(v)
        with open(record_path, 'a', encoding='utf8')as f:
            record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'model': model,
                'usage': usage_dic,
                'messages': messages.value
            }
            json.dump(record, f)
            f.write('\n')
        
    @classmethod
    def load_messages(cls, target_id=-1, print_messages=True, record_path=RECORD_PATH) -> List[str]:
        with open(record_path, 'r', encoding='utf8')as f:
            line = f.readlines()[target_id]
            messages = json.loads(line)['messages']
        if print_messages:
            print(Messages(messages))
        return messages

    @classmethod
    def calculate_usage(
        cls, 
        start_time=datetime.datetime(1970,1,1), 
        end_time=datetime.datetime(3000,1,1), 
        record_path=RECORD_PATH,
    ):
        total_usage = 0
        with open(record_path, 'r', encoding='utf8')as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not start_time <= datetime.datetime.fromisoformat(record['timestamp']) <= end_time:
                    continue
                model = record['model']
                for target, cost in usage_bill_dic[model].items():
                    total_usage += record['usage'][target]*cost
        return total_usage 


class llm_api:
    """
    content:
    - messages: call once
    - str: as user's query, call once
    - list[str]: insert `QUERY_PLACEHOLDER`, call many times
        
    """
    def __new__(
        cls,
        messages:Union[Messages, str, List[str], ],
        model:Literal['claude-3-opus-20240229', 'gpt-3.5-turbo', 'gpt-4-turbo'],
        max_retry=30,
        print_messages=False,
        record_path=RECORD_PATH,
    ) -> Messages:
        
        cur_messages = Messages()
        while cls.__prepare_messages(messages, cur_messages):
            response_str, usage = cls.__main(
                func=partial(cls.chat, model=model),
                messages=cur_messages,
                max_retry=max_retry,
            )
            cur_messages.add_bot(response_str)

            APIRecordManager.dump_record(
                model=model,
                usage=usage,
                messages=cur_messages,
                record_path=record_path,
            )

        if print_messages:
            print(cur_messages)
        return cur_messages
    
    @classmethod
    def __prepare_messages(cls, messages, cur_messages:Messages):
        if isinstance(messages, Messages):
            if len(messages) < len(cur_messages):
                return False
            else:
                cur_messages.value = dcopy(messages.value)
                return True
        elif isinstance(messages, str):
            if len(cur_messages) == 1:
                return False
            else:
                cur_messages.add_user(messages)
                return True
        elif isinstance(messages, (list, tuple)):
            for p in range(len(cur_messages.value), len(messages)):
                if messages[p] == QUERY_PLACEHOLDER:
                    return True
                if p&1:
                    cur_messages.add_bot(messages[p])
                else:
                    cur_messages.add_user(messages[p])
            return False
    
    @classmethod
    def __main(cls, func, messages:Messages, max_retry):
        for retry_time in range(1, max_retry+1):
            def retry_func():
                print(traceback.format_exc())
                print(f'{"*"*5} retry {retry_time} {"*"*5}')
                time.sleep(10)  
            
            def error_func():
                print(traceback.format_exc())
                print('='*20)
                print(messages.value)
                exit()
                
            try:
                return func(messages)
            except AttributeError:
                retry_func()
            except openai.APIConnectionError:
                retry_func()
            except openai.AuthenticationError:
                if '额度已用尽' in traceback.format_exc():
                    error_func()
                else:
                    retry_func()    
            except openai.InternalServerError:
                if '无可用渠道' in traceback.format_exc():
                    error_func()
                else:
                    retry_func()
            except KeyboardInterrupt:
                error_func()
            except:
                retry_func()
        else:
            error_func()

    @classmethod
    def chat(cls, messages:Messages, model):
        client = OpenAI(
            base_url='https://api.pumpkinaigc.online/v1',
            api_key=_API_KEY_DIC[model],
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages.value,
            # logprobs=True,
            # max_tokens=3,
        )
        response_str = response.choices[0].message.content
        usage = response.usage
        return response_str, usage
    
    @classmethod
    def embed(cls, text:str, model:Literal[''], dimensions):
        client = OpenAI(
            base_url='https://api.pumpkinaigc.online/v1',
            api_key=_API_KEY_DIC[model],
        )
        response = client.embeddings.create(
            input=text,
            model=model,
            dimensions=dimensions,
        )
        embedding = response.data[0].embedding
        usage = response.usage
        return embedding, usage
    
                
if __name__ == '__main__':
    # res = chat_api(
    #     # content='hello, what\' your name?',
    #     content=['hello, what\'s the weather today', 'maybe i should not ask you this question'],
    #     model='gpt-3.5-turbo',
    #     # model='gpt-4-turbo',
    #     show_output=True
    # )
    # print(res)
    # sample_messages = Messages()
    # sample_messages.add_user('hello')
    # sample_messages.add_bot('fuck out')
    # # sample_messages.add_user('do not be angry')
    # # sample_messages.add_bot('sorry')
    # sample_messages.add_user('why would you be so rude?')
    llm_api(
        messages=['hello', 'fuck out', 'how rude you are!', QUERY_PLACEHOLDER, QUERY_PLACEHOLDER],
        model='gpt-3.5-turbo', print_messages=True,
    )
    # res = ChatAPI(Messages('hello'), model='gpt-3.5-turbo')
    # print(res)
    # res = chat_api_template(
    #     template=['hello', QUERY_PLACEHOLDER, 'who is the president of usa', QUERY_PLACEHOLDER, QUERY_PLACEHOLDER],
    #     model='gpt-3.5-turbo',
    #     show_output=True,
    #     max_retry=10,
    # )
    # print(res)
    
    # get_content(-1, print_content=True)
    
    # print(calculate_usage())
    pass