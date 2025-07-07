from utils_zp import *

"""
https://huggingface.co/Qwen/Qwen2.5-Omni-7B

pip install accelerate
pip install qwen-omni-utils[decord] -U
"""


class QwenOmni:
    use_audio_in_video = False
    return_audio = False

    def __init__(
        self, 
        model_or_model_path='/home/zhipang/pretrained_models/Qwen2.5-Omni-7B', 
        mode:Literal['auto', 'bf16', '4bit', '8bit']='bf16', 
        input_device:Literal['auto', 'cuda:0', 'cuda:1']='auto'
    ):


        if mode == 'auto':
            model_arg_map = {
                'torch_dtype': 'auto',
                'device_map': 'auto',
            }
        elif mode == 'bf16':
            model_arg_map = {
                'torch_dtype': torch.bfloat16,
                'device_map': 'auto',
                'attn_implementation': 'flash_attention_2'
            }
        elif mode == '4bit':
            bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
            model_arg_map = {
                'torch_dtype': torch.bfloat16,
                'device_map': 'auto',
                'quantization_config': bnb_config,
                'attn_implementation': 'flash_attention_2'
            }
        elif mode == '8bit':
            bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
            model_arg_map = {
                'torch_dtype': torch.bfloat16,
                'device_map': 'auto',
                'quantization_config': bnb_config,
                'attn_implementation': 'flash_attention_2'
            }
        else:
            raise Exception(f'wrong mode: {mode}')

        self.model = None
        self.processor = None
        self.model_or_model_path = model_or_model_path
        self.model_arg_map = model_arg_map
        self.input_device = input_device

    def load_model(self):
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        if self.model is None:
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_or_model_path, 
                **self.model_arg_map
            )
        if self.processor is None:
            self.processor = Qwen2_5OmniProcessor.from_pretrained(
                self.model_or_model_path
            )

    def __call__(self, conversation, only_output_assistant:bool=True):
        from qwen_omni_utils import process_mm_info

        if not self.model or not self.processor:
            self.load_model()

        model, processor, input_device = self.model, self.processor, self.input_device
        USE_AUDIO_IN_VIDEO = self.use_audio_in_video

        # Preparation for inference
        text = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, 
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        inputs = processor(
            text=text, audio=audios, images=images, videos=videos, 
            return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        if input_device == 'auto':
            inputs = inputs.to(model.device).to(model.dtype)
        else:
            inputs = inputs.to(input_device).to(model.dtype)
        # print(inputs.device)
        # print(model.device)
        if self.return_audio:
            raise Exception("self.return_audio should not be True. TODO")
            text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        else:
            text_ids = model.generate(
                **inputs, 
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
                return_audio=False
            )

        text:List[str] = processor.batch_decode(
            text_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        if only_output_assistant:
            text = [p.split('assistant\n', 1)[1] for p in text]
        return text


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

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
                # {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
                {'type': 'text', 'text': 'what is Large Language Model, explain from different aspects. return in json format'}
            ],
        },
    ]
    print()
    print(
        # QwenOmni(input_device='cuda:1')(conversation)[0]
        QwenOmni(input_device='cuda:0')(conversation)[0]
    )