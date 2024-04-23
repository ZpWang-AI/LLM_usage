import datetime
import json
import openai
import time

from typing import *
from pathlib import Path as path
from openai import OpenAI


RECORD_ROOT_PATH = path(__file__).parent.parent/'api_record'
USEAGE_RECORD_PATH = RECORD_ROOT_PATH/'usage.jsonl'
QA_RECORD_PATH = RECORD_ROOT_PATH/'qa.jsonl'
path(USEAGE_RECORD_PATH).parent.mkdir(parents=True, exist_ok=True)
path(QA_RECORD_PATH).parent.mkdir(parents=True, exist_ok=True)


api_key = '''
claude-3-opus-20240229
sk-y6d5sFcYw0qvi6cC7678F58a26B04b58A9D3D881380e7148

gpt-3.5-turbo
sk-aKOThu0aARoVOtLOAdC4722a930e4875A2De29Cb1948301f
'''
# gpt-3.5-turbo
# sk-gU4D0bOrRgAmASyF4fFeCd76D1Fb486cA77a75190d8c42Ad
# sk-jk7440gbyWkGBBys67369542Df3f40589c834350A8Fd9687

api_key = [i.strip()for i in api_key.split() if i.strip()]
api_key_dic = dict(zip(api_key[::2], api_key[1::2]))


class Messages: 
    def __init__(self, messages:Union[List[dict], None]=None) -> None:
        if messages is not None:
            if hasattr(messages, 'messages'):
                self.messages = messages.messages
            else:
                self.messages = messages
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
    model=Literal['claude-3-opus-20240229', 'gpt-3.5-turbo'],
    max_retry=30,
    show_output=False,
) -> Union[str, List[str]]:
    # return 'test output'
    client = OpenAI(
        base_url='https://api.pumpkinaigc.online/v1',
        api_key=api_key_dic[model],
    )
    
    if messages is not None:
        messages = Messages(messages)
    else:
        messages = Messages()
        if content is None:
            raise Exception('empty input of chat_api')
        else:
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
    
    messages: Messages
    for _ in range(max_retry):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages.messages,
                # max_tokens=3,
            )
            break
        except openai.APIConnectionError:
            time.sleep(10)
        except:
            import traceback
            print(traceback.format_exc())
            print('='*20)
            print(messages.messages)
            exit()
            
    response_content = response.choices[0].message.content
    usage = response.usage
    messages.add_bot(response_content)
    
    with open(USEAGE_RECORD_PATH, 'a', encoding='utf8')as f:
        json.dump(dict(usage), f)
        f.write('\n')
    with open(QA_RECORD_PATH, 'a', encoding='utf8')as f:
        json.dump(messages.messages, f)
        # json.dump({'query': content, 'ans': response_content}, f)
        f.write('\n')
    if show_output:
        print(messages)
    return response_content


def get_content(target_line=-1, print_content=True):
    with open(QA_RECORD_PATH, 'r', encoding='utf8')as f:
        line = f.readlines()[target_line]
        messages = json.loads(line)
    if print_content:
        print(Messages(messages))
    return messages

                
if __name__ == '__main__':
    res = chat_api(
        # content='hello, what\' your name?',
        content=['hello, what\'s the weather today', 'maybe i should not ask you this question'],
        model='gpt-3.5-turbo',
        show_output=True
    )
    print(res)
    
    # get_content(-1, print_content=True)