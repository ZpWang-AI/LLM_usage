import json
import datetime

from typing import *
from pathlib import Path as path


class ReasonFrame(dict):
    def __init__(
        self,
        prompt,
        version:str,
        data_name:Literal['pdtb2', 'pdtb3', 'conll'],
        label_level:Literal['level1', 'level2', 'raw'],
        relation:Literal['Implicit', 'Explicit', 'All'],
        model:str,    
        create_time:str=None,
    ) -> None:
        self.prompt = prompt
        self.version = version
        self.data_name = data_name
        self.label_level = label_level
        self.relation = relation
        self.model = model
        if not create_time:
            self.create_time = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.create_time = create_time
        
    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__dict__[__name] = __value
        self[__name] = __value
            
    @staticmethod
    def load_json(json_path):
        with open(json_path, 'r', encoding='utf8')as f:
            content = json.load(f)
        return ReasonFrame(**content)
    
    def dump_json(self, prompt_space):
        dump_path = path(prompt_space)/f'{self.create_time}.{self.model}.{self.version}.json'
        if not dump_path.exists():
            with open(dump_path, 'w', encoding='utf8')as f:
                json.dump(self, f, indent=2)
            print(f'dump {self.__class__.__name__} to\n{dump_path}\n')
    
    def get_reason_data(self, data_path=None):
        if not data_path:
            data_path = path(
                '/public/home/hongy/zpwang/LLaMA-Factory/zpwang/data/reason',
                f'R_{self.version}.jsonl'
            )
        with open(data_path, 'r', encoding='utf8')as f:
            reason_data = [json.loads(line)['reason']for line in f.readlines()]        
        return reason_data
