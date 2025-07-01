from utils_zp import *

"""
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

pip install accelerate
pip install qwen_vl_utils
"""


class QwenVL:
    def __init__(
        self, 
        model_or_model_path='/home/zhipang/pretrained_models/Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5', 
        # mode:Literal['auto', 'bf16', '4bit', '8bit']='bf16', 
        mode:Literal['auto']='auto', 
        input_device:Literal['auto', 'cuda:0', 'cuda:1']='auto',
    ):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        if mode == 'auto':
            model_arg_map = {
                'torch_dtype': 'auto',
                'device_map': 'auto',
            }
        # elif mode == 'bf16':
        #     model_arg_map = {
        #         'torch_dtype': torch.bfloat16,
        #         'device_map': 'auto',
        #         'attn_implementation': 'flash_attention_2'
        #     }
        # elif mode == '4bit':
        #     bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')
        #     model_arg_map = {
        #         'torch_dtype': torch.bfloat16,
        #         'device_map': 'auto',
        #         'quantization_config': bnb_config,
        #         'attn_implementation': 'flash_attention_2'
        #     }
        # elif mode == '8bit':
        #     bnb_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        #     model_arg_map = {
        #         'torch_dtype': torch.bfloat16,
        #         'device_map': 'auto',
        #         'quantization_config': bnb_config,
        #         'attn_implementation': 'flash_attention_2'
        #     }
        else:
            raise Exception(f'wrong mode: {mode}')

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_or_model_path, 
            **model_arg_map
        )
        self.processor = AutoProcessor.from_pretrained(
            model_or_model_path,
            use_fast=True,
        )
        self.input_device = input_device

    def __call__(self, conversation, only_output_assistant:bool=True):
        from qwen_vl_utils import process_vision_info

        model, processor, input_device = self.model, self.processor, self.input_device

        # Preparation for inference
        text = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            tokenize=False
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if input_device == 'auto':
            inputs = inputs.to(model.device).to(model.dtype)
        else:
            inputs = inputs.to(input_device).to(model.dtype)

        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text:List[str] = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # if only_output_assistant:
        #     text = [p.split('assistant\n', 1)[1] for p in text]
        return output_text


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5,6,7'

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
                {
                    "type": "video", 
                    "video": "file:///home/zhipang/PhysicalDynamics/data/Annotation/pipeline.wisa_v3_2.example/yes/3/1d9c24ec6d0d48b33ef4765edc267a483993eada518649a29620c5861f9bfe13.mp4",
                    "max_pixels": 512 * 512,
                    "fps": 5.0,
                },
                {'type': 'text', 'text': '''## Objective
Identify the dominant subjects in the video.

## Guidelines
* List subjects with concise noun phrases, one per line.
* Prioritize subjects that are active, dynamic, or central to the scene.
* Ignore static/background objects (e.g., trees, furniture) and minor/secondary elements.
* If there are no dominant subjects, simply output "None".

## Output Examples
### Example 1
woman in red dress
golden pet dog

### Example 2
burning campfire
ice cube

### Example 3
None'''}
            ],
        },
    ]
    print()
    print(
        QwenVL()(conversation)[0]

    )