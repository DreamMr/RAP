from rap.config import supported_VLM
import os
import traceback
import time
from PIL import Image
import time

model = supported_VLM['llava_v1.5_7b'](max_new_tokens=1024,debug=True,is_process_image=False,processed_image_path=r'/mnt/code/users/wangwenbin/LVLM/Evaluation/official/VLMEvalKit/debug_image',max_step=20, rag_model_path=r'/mnt/data/users/wenbinwang/huggingface/VisRAG-Ret')
image_path = r'/mnt/data/users/wenbinwang/datasets/LVLM_Benchmark/LMUData/images/hr_bench_8k/60.jpg'
while True:
    try:
        cur_input = """What is the color of the umbrella?
A. Blue
B. Black
C. Green
D. Red
Answer the option letter directly."""
#         cur_input = """What is the title of the framed poster visible in the image?
        
# Options:
# A. Ely Diocese
# B. Ely Division
# C. Ely Diocess
# D. Ely Cathedral
# Please select the correct answer from the options above."""
#         cur_input = """Where is the small stone cairn located relative to the waterfall?
        
# Options:
# A. At the bottom right of the waterfall
# B. In the middle of the waterfall
# C. At the top of the waterfall
# D. To the left of the waterfall
# Please select the correct answer from the options above.""" # 308
        if cur_input == 'QUIT':
            break
        elif cur_input == 'CHANGE_IMAGE':
            image_path = input("Please enter image path: ")
            continue
        
        start_time = time.time()
        ret = model.generate([dict(type='image',value=image_path),
                                dict(type='text',value=cur_input)
                ], no_rag=False)
        print("Cost time: {} sec".format((time.time() - start_time)))
        print("##### Response ######")
        print(ret)
        break
    except Exception as e:
        print(e)
        traceback.print_exc()


