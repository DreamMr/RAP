from rap.vlm import *
from rap.api import *
from functools import partial

llava_series = {
    'llava_onevision_qwen2_0.5b_ov': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-0.5b-ov'),
    'llava_v1.5_7b': partial(LLaVA, model_path='liuhaotian/llava-v1.5-7b'),
    'llava_v1.5_13b': partial(LLaVA, model_path='liuhaotian/llava-v1.5-13b'),
}

qwen3_series = {
    "Qwen3-VL-8B-Instruct": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-8B-Instruct",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0., 
        max_new_tokens=10,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
}

supported_VLM = {}

model_groups = [
    llava_series, qwen3_series
]

for grp in model_groups:
    supported_VLM.update(grp)
