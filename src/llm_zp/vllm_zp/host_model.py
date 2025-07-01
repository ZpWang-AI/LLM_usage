import os, subprocess
from typing import *


def host_model(
    model:str,
    devices:Iterable[int]=None,
):
    if devices:
        devices = list(map(int, devices))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        subprocess.run(
            [
                "vllm", "serve", str(model),
                "--tensor-parallel-size", str(len(devices))
            ],
            env=env,  # 传递环境变量
            # check=True
        )
    else:
        subprocess.run([
            'vllm', 'serve', str(model)
        ],)


if __name__ == '__main__':
    host_model(
        'Qwen/Qwen2.5-VL-7B-Instruct',
        [4,5,6,7]
    )
    