import json
import datetime
from pathlib import Path as path


class AttrDict(dict):
    def __setattr__(self, __name: str, __value) -> None:
        if type(__value) == path:
            raise Exception('attribution type can\'t be "Path"')
        self.__dict__[__name] = __value
        self[__name] = __value
        
    def set_create_time(self, create_time=None):
        if not create_time:
            self.create_time = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.create_time = create_time

    def __dump_json(self, json_path, overwrite=True):
        json_path = path(json_path)
        if not json_path.exists() or overwrite:
            with open(json_path, 'w', encoding='utf8')as f:
                json.dump(self, f, indent=4, ensure_ascii=False)
    
    def __load_json(self, json_path, default:dict=None):
        json_path = path(json_path)
        if not json_path.exists():
            if default:
                return self.__class__(**default)
            else:
                raise Exception('file does not exists')
        else:
            with open(json_path, 'r', encoding='utf8')as f:
                dic = json.load(f)
            return self.__class__(**dic)
