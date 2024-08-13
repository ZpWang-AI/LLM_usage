import datetime
import json
import openai
import time
import traceback

from typing import *
from pathlib import Path as path
from openai import OpenAI


RECORD_ROOT_PATH = path(__file__).parent.parent/'api_record'
RECORD_PATH = RECORD_ROOT_PATH/'record.jsonl'
path(RECORD_PATH).parent.mkdir(parents=True, exist_ok=True)


api_key = '''
claude-3-opus-20240229
sk-y6d5sFcYw0qvi6cC7678F58a26B04b58A9D3D881380e7148

gpt-3.5-turbo
sk-nOjUfMVeQiAWCO9nA5E4EaBfFeB54b9a8d2985FaEe30F8Fe

gpt-4-turbo
sk-cCFxhEZgnjm42hfoCb9632Ff8d6b48CfA2743a0e9cAe0d33
'''
# sk-vlnPP3ywKrwypDR863Ac9aF2Ab964fA18fC8D08dF5D81552 gpt4
usage_bill_dic = {
    'gpt-3.5-turbo': {'prompt_tokens': 1.5e-6, 'completion_tokens': 4.5e-6,},
    'claude-3-opus-20240229': {'prompt_tokens': 15e-6, 'completion_tokens': 75e-6,},
    'gpt-4-turbo': {'prompt_tokens': 10e-6, 'completion_tokens': 30e-6,},
}

api_key = [i.strip()for i in api_key.split() if i.strip()]
api_key_dic = dict(zip(api_key[::2], api_key[1::2]))


class Messages: 
    def __init__(self, messages:Union[List[dict], 'Messages', None]=None) -> None:
        if messages is not None:
            if hasattr(messages, 'messages'):
                self.messages = messages.messages
            elif isinstance(messages, list) and all(isinstance(p, dict)for p in messages):
                self.messages = messages
            else:
                print(messages)
                raise Exception('wrong type of messages')
        else:
            self.messages = []

    def __add_content(self, role, content):
        self.messages.append({'role': role, 'content': content})
        
    def add_user(self, content):
        self.__add_content('user', content)
        
    def add_bot(self, content):
        self.__add_content('assistant', content)

    def add_system(self, content):
        self.__add_content('system', content)
        
    def __repr__(self) -> str:
        res = []
        for piece in self.messages:
            res.append(f'=== {piece["role"]} ===')
            res.append(piece['content']+'\n')
        res.append('='*20)
        return '\n'.join(res)
    

def chat_api(
    content:Union[str, Iterable[str]]=None,
    messages:Union[Messages, List[dict]]=None,
    model=Literal['claude-3-opus-20240229', 'gpt-3.5-turbo', 'gpt-4-turbo'],
    max_retry=30,
    show_output=False,
) -> Union[str, List[str]]:
    """
    if content is str, return str
    if content is List[str], return List[str]
    the same as messages input
    """
    # prepare messages
    if messages is None:
        if content is None:
            raise Exception('empty input of chat_api')
        else:
            messages = Messages()
            if isinstance(content, str):
                messages.add_user(content)

            elif hasattr(content, '__iter__'):
                for pstr in content:
                    messages.add_user(pstr)
                    chat_api(
                        messages=messages,
                        model=model, max_retry=max_retry,
                        show_output=False
                    )
                if show_output:
                    print(messages)
                return [p['content']for p in messages.messages 
                        if p['role'] == 'assistant']

            else:
                raise Exception('wrong type of chat_api')

    else:
        messages = Messages(messages)
    
    # chat
    client = OpenAI(
        base_url='https://api.pumpkinaigc.online/v1',
        api_key=api_key_dic[model],
    )
    messages: Messages
    for retry_time in range(1, max_retry+1):
        def retry_func():
            print(traceback.format_exc())
            print(f'{"*"*5} retry {retry_time} {"*"*5}')
            time.sleep(10)  
        
        def error_func():
            print(traceback.format_exc())
            print('='*20)
            print(messages.messages)
            exit()
            
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages.messages,
                # max_tokens=3,
            )
            response_content = response.choices[0].message.content
            usage = response.usage
            messages.add_bot(response_content)
            break
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
    
    # record
    with open(RECORD_PATH, 'a', encoding='utf8')as f:
        record = {
            'timestamp': time.time(),
            'model': model,
            'usage': dict(usage),
            'messages': messages.messages
        }
        json.dump(record, f)
        f.write('\n')
    if show_output:
        print(messages)
    return response_content


def get_content(target_line=-1, print_content=True):
    with open(RECORD_PATH, 'r', encoding='utf8')as f:
        line = f.readlines()[target_line]
        messages = json.loads(line)['messages']
    if print_content:
        print(Messages(messages))
    return messages


def calculate_usage(record_path=RECORD_PATH, start_timestamp=-1, end_timestamp=float('inf')):
    total_usage = 0
    with open(record_path, 'r', encoding='utf8')as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not start_timestamp <= record['timestamp'] <= end_timestamp:
                continue
            model = record['model']
            for target, cost in usage_bill_dic[model].items():
                total_usage += record['usage'][target]*cost
    return total_usage 

                
if __name__ == '__main__':
    res = chat_api(
        # content='hello, what\' your name?',
        content=['hello, what\'s the weather today', 'maybe i should not ask you this question'],
        # model='gpt-3.5-turbo',
        model='gpt-4-turbo',
        show_output=True
    )
    print(res)
    
    # get_content(-1, print_content=True)
    
    # print(calculate_usage())
    pass