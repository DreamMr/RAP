from rap.vlm import *
from rap.api import *
from functools import partial


LLAVA_V1_7B_MODEL_PTH = 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '

llava_series = {
    'llava_onevision_qwen2_0.5b_ov': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-0.5b-ov'),
}

supported_VLM = {}

model_groups = [
    llava_series
]

for grp in model_groups:
    supported_VLM.update(grp)
