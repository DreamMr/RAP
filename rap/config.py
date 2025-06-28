from rap.vlm import *
from rap.api import *
from functools import partial

llava_series = {
    'llava_onevision_qwen2_0.5b_ov': partial(LLaVA_OneVision, model_path='lmms-lab/llava-onevision-qwen2-0.5b-ov'),
}

supported_VLM = {}

model_groups = [
    llava_series
]

for grp in model_groups:
    supported_VLM.update(grp)
