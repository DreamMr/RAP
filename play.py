from rap.config import supported_VLM
import os
import traceback
import time
from PIL import Image
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llava_onevision_qwen2_0.5b_ov')
    parser.add_argument('--model_path', type=str, default="lmms-lab/llava-onevision-qwen2-0.5b-ov")
    parser.add_argument('--rag_model_path', type=str, default="openbmb/VisRAG-Ret")
    parser.add_argument('--image_path', type=str, default="./demo.jpg")
    parser.add_argument('--use_rap', action='store_true')
    parser.add_argument('--input', type=str, required=True)
    args = parser.parse_args()

    model = supported_VLM[args.model](model_path=args.model_path, max_new_tokens=1024,debug=False,is_process_image=False,processed_image_path=r'./outputs',max_step=10, rag_model_path=args.rag_model_path)
    image_path = args.image_path
    cur_input = args.input
    use_rap = not args.use_rap
    ret = model.generate([dict(type='image',value=image_path),
                                dict(type='text',value=cur_input)
                ], no_rag=use_rap)

    print(f"Image_path: {image_path}, input: {cur_input}\nResponse: {ret}")