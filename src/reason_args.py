import json
import datetime

from typing import *
from pathlib import Path as path

from utils import AttrDict


class ReasonArgs(AttrDict):
    def __init__(
        self,
        prompt,
        version:str,
        data_name:Literal['pdtb2', 'pdtb3', 'conll'],
        label_level:Literal['level1', 'level2', 'raw'],
        relation:Literal['Implicit', 'Explicit', 'All'],
        model:str,    
        create_time=None,
    ) -> None:
        self.prompt = prompt
        self.version = version
        self.data_name = data_name
        self.label_level = label_level
        self.relation = relation
        self.model = model
        self.set_create
        if not create_time:
            self.create_time = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.create_time = create_time