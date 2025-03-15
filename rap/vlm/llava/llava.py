import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE
import copy


class LLaVA(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='liuhaotian/llava_v1.5_7b', debug=False,
                 is_process_image=False, processed_image_path=None, max_step=200, bias_value=0.2, rag_model_path='openbmb/VisRAG-Ret',
                 **kwargs):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except:
            warnings.warn('Please install llava before using LLaVA')
            sys.exit(-1)
        super().__init__(debug=debug, is_process_image=is_process_image, processed_image_path=processed_image_path, max_step=max_step, bias_value=bias_value, rag_model_path=rag_model_path)
        warnings.warn('Please install the latest version of llava from github before you evaluate the LLaVA model. ')
        assert osp.exists(model_path) or splitlen(model_path) == 2
        self.system_prompt = (
            'A chat between a curious user and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        )
        self.stop_str = '</s>'

        if model_path == 'Lin-Chen/ShareGPT4V-7B':
            model_name = 'llava-v1.5-7b'
        elif model_path == 'Lin-Chen/ShareGPT4V-13B':
            model_name = 'llava-v1.5-13b'
        else:
            model_name = get_model_name_from_path(model_path)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device_map='cpu'
            )
        except:
            if 'ShareGPT4V' in model_path:
                import llava
                warnings.warn(
                    'Please manually remove the encoder type check in '
                    f'{llava.__path__[0]}/model/multimodal_encoder/builder.py '
                    'Line 8 to use the ShareGPT4V model. ')
            else:
                warnings.warn('Unknown error when loading LLaVA model.')
            exit(-1)

        self.model = self.model.cuda()
        self.conv_mode = 'llava_v1'
        self.device = self.model.device
        self.model_path = model_path

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True) # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        self.init_index_yes_no()

    def init_index_yes_no(self):
        if len(self.tokenizer("Yes").input_ids) == 1 and len(self.tokenizer("No").input_ids) == 1: 
            self.index_yes = self.tokenizer("Yes").input_ids[0]
            self.index_no = self.tokenizer("No").input_ids[0]
        else:
            assert len(self.tokenizer("Yes").input_ids) == 2 and len(self.tokenizer("No").input_ids) == 2
            self.index_yes = self.tokenizer("Yes").input_ids[1]
            self.index_no = self.tokenizer("No").input_ids[1]
    
    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False
    
    @torch.no_grad()
    def get_confidence_value(self, content, image_pil: Image.Image):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX
        image_list = [image_pil]
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        if image_pil is None:
            image_tensor = None
        else:
            image_tensor = process_images(image_list, self.image_processor, args).to(dtype=torch.float16)
            image_tensor = image_tensor.cuda(self.device)
            content = " <image> " + content
        prompt = self.system_prompt + 'USER: ' + content + ' ASSISTANT: '
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda(self.device)
        outputs = self.model(
            input_ids,
            images=image_tensor,
            return_dict=True
        )
        return self._cal_confidence(outputs)
    
    @torch.no_grad()
    def _cal_confidence(self, outputs):
        logits_yesno = outputs.logits[0, -1, [self.index_yes, self.index_no]]
        confidence = torch.softmax(logits_yesno, dim=-1)[0] 
        confidence = 2 * (confidence.item()) # [0, 2] align with retrieval score
        return confidence
    

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer the option letter directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def concat_tilist(self, message):
        text, images = '', []
        for item in message:
            if item['type'] == 'text':
                text += item['value']
            elif item['type'] == 'image':
                text += '<image>\n'
                images.append(item['value'])
        return text, images

    def chat_inner(self, message, dataset=None):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX

        prompt = self.system_prompt
        images = []
        for utter in message:
            prompt += 'USER: ' if utter['role'] == 'user' else 'ASSISTANT: '
            content, images_sub = self.concat_tilist(utter['content'])
            prompt += content
            images.extend(images_sub)
            prompt += ' ' if utter['role'] == 'user' else self.stop_str
        assert message[-1]['role'] == 'user', message
        prompt += 'ASSISTANT: '

        images = [Image.open(s).convert('RGB') for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images(images, self.image_processor, args).to('cuda', dtype=torch.float16)

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output

    def generate_inner(self, message, dataset=None):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX

        # Support interleave text and image
        content, images = self.concat_tilist(message)

        images = [Image.open(s).convert('RGB') if isinstance(s,str) else s for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        if images:
            image_tensor = process_images(images, self.image_processor, args).to(dtype=torch.float16)
            image_tensor = image_tensor.cuda(self.model.device)
        else:
            image_tensor = None

        prompt = self.system_prompt + 'USER: ' + content + ' ASSISTANT:'
        print(prompt)
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda(self.model.device)
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output


class LLaVA_OneVision(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = '<image>'
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path='lmms-lab/llava-onevision-qwen2-7b-si', debug=False,
                 is_process_image=False, processed_image_path=None,max_step=200, bias_value=0.2, rag_model_path='openbmb/VisRAG-Ret',**kwargs):
        assert model_path is not None
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates
            from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        except ImportError:
            warnings.warn('Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`')
        super().__init__(debug=debug, is_process_image=is_process_image, processed_image_path=processed_image_path,max_step=max_step, bias_value=bias_value, rag_model_path=rag_model_path)
        model_name = get_model_name_from_path(model_path)
        self.model_path = model_path
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, device_map=None)
        model.cuda().eval()
        model.tie_weights()

        if 'llava' in model_path.lower():
            conv_mode = 'qwen_1_5'
        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = process_images  # Store process_images as a class attribute
        self.device = self.model.device
        self.init_index_yes_no()
        
    def init_index_yes_no(self):
        if len(self.tokenizer("Yes").input_ids) == 1 and len(self.tokenizer("No").input_ids) == 1: 
            self.index_yes = self.tokenizer("Yes").input_ids[0]
            self.index_no = self.tokenizer("No").input_ids[0]
        else:
            assert len(self.tokenizer("Yes").input_ids) == 2 and len(self.tokenizer("No").input_ids) == 2
            self.index_yes = self.tokenizer("Yes").input_ids[1]
            self.index_no = self.tokenizer("No").input_ids[1]
        
    @torch.no_grad()
    def get_confidence_value(self, content, image_list: Image.Image):
        if not isinstance(image_list, list):
            image_list = [image_list]
        image_sizes = [img.size for img in image_list]
        image_tensor = self.process_images(image_list, self.image_processor, self.model.config).to(dtype=torch.float16)
        image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]
        content = self.DEFAULT_IMAGE_TOKEN + '\n' + content
        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda(self.device)
        outputs = self.model(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            modalities=["image"] * len(input_ids),
            return_dict=True
        )
        return self._cal_confidence(outputs)
    
    @torch.no_grad()
    def _cal_confidence(self, outputs):
        logits_yesno = outputs.logits[0, -1, [self.index_yes, self.index_no]]
        confidence = torch.softmax(logits_yesno, dim=-1)[0] 
        confidence = 2 * (confidence.item() - 0.5) # [-1, 1]
        return confidence

    def generate_inner(self, message, dataset=None):
        content, images = '', []
        image_sizes = []  # Store image sizes

        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                if isinstance(msg['value'],str):
                    img = Image.open(msg['value']).convert('RGB')
                else:
                    img = msg['value']
                images.append(img)
                image_sizes.append(img.size)  # Store the size of each image
                content = self.DEFAULT_IMAGE_TOKEN + '\n' + content

        # Process images using the class attribute self.process_images
        image_tensor = self.process_images(images, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(prompt_question,
                                               self.tokenizer,
                                               self.IMAGE_TOKEN_INDEX,
                                               return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda(self.device)

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs
