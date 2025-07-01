import requests
from openai import OpenAI


class ModelAPI:
    def __init__(self, model, url, api_key:str='1'):
        self.model = model
        self.base_url = url.rstrip('/')
        self.api_key = api_key
        self.client = OpenAI(base_url=url, api_key=api_key)

    def __call__(self, conversation):
        # completion = self.client.beta.chat.completions.parse(
        #     model=self.model, messages=conversation
        # )
        completion = self.client.chat.completions.create(
            model=self.model, 
            messages=conversation,
            response_format={'type': 'json_object'}
        )
        return completion.choices[0].message.content

    # def __call__(self, conversation):
    #     endpoint = f"{self.base_url}/chat/completions"
    #     headers = {
    #         "Authorization": f"Bearer {self.api_key}",
    #         "Content-Type": "application/json",
    #     }
    #     payload = {
    #         "model": self.model,
    #         "messages": conversation,
    #         "response_format": {"type": "json_object"}
    #     }
        
    #     response = requests.post(endpoint, headers=headers, json=payload)
    #     response.raise_for_status()  # Raise error for HTTP failures
    #     result = response.json()
    #     return result['choices'][0]['message']['content']

if __name__ == '__main__':
    _api = ModelAPI(
        'Qwen/Qwen2.5-VL-7B-Instruct',
        'http://0.0.0.0:8000/v1',
    )
    
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {'type': 'text', 'text': 'what are the main subjects in the video? return list in json format.'},
                {
                    "type": "video",
                    "video": "file:///home/zhipang/PhysicalDynamics/data/Annotation/pipeline.wisa_v3_2.example/yes/3/1d9c24ec6d0d48b33ef4765edc267a483993eada518649a29620c5861f9bfe13.mp4",
                    # 'image': 'file:///home/zhipang/PhysicalDynamics/src/~yolo_sample/frame_000000.jpg',
                    # 'video': [
                    #     'file:///home/zhipang/PhysicalDynamics/src/~yolo_sample/frame_000000.jpg',
                    #     'file:///home/zhipang/PhysicalDynamics/src/~yolo_sample/frame_000001.jpg',
                    #     'file:///home/zhipang/PhysicalDynamics/src/~yolo_sample/frame_000002.jpg',
                    #     'file:///home/zhipang/PhysicalDynamics/src/~yolo_sample/frame_000003.jpg',
                    #     # ''
                    # ],
                    # "video": [
                    #     "file://~/PhysicalDynamics/data/Annotation/pipeline.wisa_v3_2.example/yes/3/1d9c24ec6d0d48b33ef4765edc267a483993eada518649a29620c5861f9bfe13.mp4"
                    # ],
                    # "max_pixels": 360 * 420,
                    "fps": 2.0,
                },
            ],
        },
    ]
    print(_api(conversation))