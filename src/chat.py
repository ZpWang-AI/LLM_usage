import datetime
import json
import openai
import time

from typing import *
from pathlib import Path as path
from openai import OpenAI


USEAGE_RECORD_PATH = '/public/home/hongy/zpwang/LLM_API/api_record/usage.jsonl'
QA_RECORD_PATH = '/public/home/hongy/zpwang/LLM_API/api_record/qa.jsonl'
path(USEAGE_RECORD_PATH).parent.mkdir(parents=True, exist_ok=True)
path(QA_RECORD_PATH).parent.mkdir(parents=True, exist_ok=True)


api_key = '''
claude-3-opus-20240229
sk-y6d5sFcYw0qvi6cC7678F58a26B04b58A9D3D881380e7148

gpt-3.5-turbo
sk-gU4D0bOrRgAmASyF4fFeCd76D1Fb486cA77a75190d8c42Ad
'''
# gpt-3.5-turbo
# sk-jk7440gbyWkGBBys67369542Df3f40589c834350A8Fd9687

api_key = [i.strip()for i in api_key.split() if i.strip()]
api_key_dic = dict(zip(api_key[::2], api_key[1::2]))


def chat(
    content=None,
    messages=None,
    model=Literal['claude-3-opus-20240229', 'gpt-3.5-turbo'],
    max_retry=30,
):
    # return 'test output'
    client = OpenAI(
        base_url='https://api.pumpkinaigc.online/v1',
        api_key=api_key_dic[model],
    )
    if not content and not messages:
        raise Exception('empty input of chat')
    if not messages:
        messages = [{'role': 'user', 'content': content}]
    else:
        raise Exception('Do not use message. Waiting for updating.')
    
    for _ in range(max_retry):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                # max_tokens=3,
            )
            break
        except openai.APIConnectionError:
            time.sleep(10)
            
    response_content = response.choices[0].message.content
    usage = response.usage
    
    with open(USEAGE_RECORD_PATH, 'a', encoding='utf8')as f:
        json.dump(dict(usage), f)
        f.write('\n')
    with open(QA_RECORD_PATH, 'a', encoding='utf8')as f:
        json.dump({'query': content, 'ans': response_content}, f)
        f.write('\n')
    return response_content


def get_content(target_line=-1, print_content=True):
    with open(QA_RECORD_PATH, 'r', encoding='utf8')as f:
        line = f.readlines()[target_line]
        content = json.loads(line)
    if print_content:
        print('-'*20)
        print(content['query'])
        print('-'*20)
        print(content['ans'])
        print('-'*20)
    return content

                
if __name__ == '__main__':
    # res = chat(
    #     content='hello, what\' your name?',
    #     model='gpt-3.5-turbo'
    # )
    # print(res)
    
    get_content(-1, print_content=True)