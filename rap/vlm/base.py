from ..smp import *
from abc import abstractmethod
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.util import view_as_windows
import numpy as np
import math
import time
from copy import deepcopy

ANSWER_PROMPT = "Question: {}\nCould you answer the question based on the available visual information? Answer Yes or No."


def unweighted_mean_pooling(hidden, attention_mask):
    reps = torch.mean(hidden,dim=1).float()
    return reps

@torch.no_grad()
def encode(text_or_image_list, tokenizer, rag_model):
    
    max_batch_size = 64
    embedding_list = []
    for i in range(0, len(text_or_image_list), max_batch_size):
        sub_text_or_image_list = text_or_image_list[i:i+max_batch_size]
    
        if (isinstance(sub_text_or_image_list[0], str)):
            inputs = {
                "text": sub_text_or_image_list,
                'image': [None] * len(sub_text_or_image_list),
                'tokenizer': tokenizer
            }
        else:
            inputs = {
                "text": [''] * len(sub_text_or_image_list),
                'image': sub_text_or_image_list,
                'tokenizer': tokenizer
            }
        outputs = rag_model(**inputs)
        attention_mask = outputs.attention_mask
        hidden = outputs.last_hidden_state

        reps = unweighted_mean_pooling(hidden, attention_mask)   
        embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
        embedding_list.append(embeddings)
    
    embeddings = np.concatenate(embedding_list)
    return embeddings

def split_image(image, crop_size):
    width, height = image.size    
    rest_w = crop_size - width % crop_size if width % crop_size > 0 else 0
    rest_h = crop_size - height % crop_size if height % crop_size > 0 else 0
    new_width = width + rest_w
    new_height = height + rest_h
    
    new_image = Image.new('RGB', (new_width, new_height), (0,0,0))
    new_image.paste(image, (0,0))
    
    image_array = np.array(new_image)
    height, width, channels = image_array.shape
    
    patches = view_as_windows(
        image_array,
        window_shape=(crop_size, crop_size,channels),
        step=crop_size
    )
    rows, cols, _, _, _, _ = patches.shape
    image_patch_list = patches.reshape(-1, crop_size, crop_size, channels)
    
    image_list = [Image.fromarray(patch) for patch in image_patch_list]
    return image_list, rows, cols

def compress_matrix_with_indices(matrix):
    rows_to_keep = [i for i, row in enumerate(matrix) if any(cell > 0 for cell in row)]
    
    if matrix is not None and matrix.sum() > 0:
        num_cols = len(matrix[0])
        cols_to_keep = [j for j in range(num_cols) if any(matrix[i][j] > 0 for i in range(len(matrix)))]
    else:
        cols_to_keep = []
    
    compressed_matrix = []

    for compressed_row_idx, original_row_idx in enumerate(rows_to_keep):
        compressed_row = []
        for compressed_col_idx, original_col_idx in enumerate(cols_to_keep):
            value = matrix[original_row_idx][original_col_idx]
            compressed_row.append(value.item())
        compressed_matrix.append(compressed_row)
    
    return torch.tensor(compressed_matrix)

class TreeNode:
    def __init__(self, answer_score=-1, retrieval_score=-1, depth=1, confidence=-1., image=None, parent=None, select_idx=None):
        self.depth = depth
        self.confidence = confidence
        self.image = image
        self.answer_score = answer_score
        self.retrieval_score = retrieval_score
        self.parent = parent
        self.select_idx = select_idx
        self.path_name = None
        

def get_confidence_weight(cur_depth, bias_value=0.2):
    coeff = 1 - bias_value
    return coeff * (1- (1 / cur_depth)**2) + bias_value

class BaseModel:

    INTERLEAVE = False
    allowed_types = ['text', 'image', 'video','type']

    def __init__(self, debug=False, is_process_image=False, processed_image_path=None, max_step=200, bias_value=0.2, rag_model_path='openbmb/VisRAG-Ret'):
        
        if is_process_image and processed_image_path is None:
            raise ValueError("The is_process_image=True, however processed_image_path is None!!!")
        
        self.dump_image_func = None
        
        # initialize rag model
        self.rag_model_path = rag_model_path
        self.rag_tokenizer = AutoTokenizer.from_pretrained(self.rag_model_path, trust_remote_code=True)
        self.rag_model = AutoModel.from_pretrained(self.rag_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
        self.rag_model.eval()
        self.rag_image_size = 224 # 4k 224 8k:336
        self.debug=debug
        self.is_process_image = is_process_image
        self.processed_image_path = processed_image_path
        self.mapping = None
        self.max_step=max_step # 4k 100 8k 200 2k 30
        self.bias_value = bias_value
        print("Max step: {}, bias_value: {}".format(max_step, bias_value))
        if self.processed_image_path is not None:
            os.makedirs(self.processed_image_path, exist_ok=True)
        
            mapping_image_dataset = os.path.join(self.processed_image_path, 'mapping.json')
            if os.path.exists(mapping_image_dataset):
                self.mapping = load(mapping_image_dataset)
                print("load mapping: {}".format(mapping_image_dataset))
        
    def _erosion(self, image_matrix, retrieval_score, k_list, chunk_size):
        # 
        rows, cols = image_matrix.shape
        non_zero_patch_num = (image_matrix > 0).sum()
        flatten_image_matrix = torch.flatten(image_matrix).int()
        scores = retrieval_score[flatten_image_matrix]
        original_matrix = torch.zeros((rows,cols),dtype=torch.int)
        new_image_list = []
        for k in k_list:
            select_k = int(k * non_zero_patch_num)
            topk_values, topk_indices = torch.topk(scores, max(1,select_k), dim=0, largest=True, sorted=False)
            for selected_idx in topk_indices:
                #original_idx = filtered_mapping_indices[selected_idx.item()]
                row_idx = selected_idx // cols
                col_idx = selected_idx % cols
                original_matrix[row_idx][col_idx] = flatten_image_matrix[selected_idx]
            try:
                compressed_matrix = compress_matrix_with_indices(original_matrix)
            except Exception as e:
                print(e)
                print(compressed_matrix.shape)
                print(original_matrix.shape)
            new_image_list.append(compressed_matrix)
        return new_image_list

    def use_custom_prompt(self, dataset):
        """Whether to use custom prompt for the given dataset.

        Args:
            dataset (str): The name of the dataset.

        Returns:
            bool: Whether to use custom prompt. If True, will call `build_prompt` of the VLM to build the prompt.
                Default to False.
        """
        return False

    @abstractmethod
    def build_prompt(self, line, dataset):
        """Build custom prompts for a specific dataset. Called only if `use_custom_prompt` returns True.

        Args:
            line (line of pd.DataFrame): The raw input line.
            dataset (str): The name of the dataset.

        Returns:
            str: The built message.
        """
        raise NotImplementedError

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    @abstractmethod
    def generate_inner(self, message, dataset=None):
        raise NotImplementedError

    def check_content(self, msgs):
        """Check the content type of the input. Four types are allowed: str, dict, liststr, listdict.
        """
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'

    def preproc_content(self, inputs):
        """Convert the raw input messages to a list of dicts.

        Args:
            inputs: raw input messages.

        Returns:
            list(dict): The preprocessed input messages. Will return None if failed to preprocess the input.
        """
        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif self.check_content(inputs) == 'dict':
            assert 'type' in inputs and 'value' in inputs
            return [inputs]
        elif self.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                mime, pth = parse_file(s)
                if mime is None or mime == 'unknown':
                    res.append(dict(type='text', value=s))
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))
            return res
        elif self.check_content(inputs) == 'listdict':
            for item in inputs:
                assert 'type' in item and 'value' in item
                mime, s = parse_file(item['value'])
                if mime is None:
                    assert item['type'] == 'text'
                else:
                    assert mime.split('/')[0] == item['type']
                    item['value'] = s
            return inputs
        else:
            return None
        
    def create_image_memory(self, image, chunk_size):
        image_patch_list, rows, cols = split_image(image, chunk_size)
        
        image_matrix = torch.zeros(rows, cols,dtype=torch.int)
        for idx in range(len(image_patch_list)):
            image_matrix[idx // cols][idx % cols] = idx + 1
            
        image_patch_list = [Image.new('RGB',(chunk_size, chunk_size))] + image_patch_list
        embedding_image = encode(image_patch_list, self.rag_tokenizer, self.rag_model)
        
        return embedding_image, image_matrix, image_patch_list
    
    def map2image(self, image_matrix, image_patch_list, chunk_size):
        rows, cols = image_matrix.shape
        new_image = Image.new('RGB', (cols * chunk_size, rows * chunk_size))
        for i in range(rows):
            for j in range(cols):
                x = i * chunk_size
                y = j * chunk_size
                new_image.paste(image_patch_list[int(image_matrix[i][j])], (y,x))
        return new_image
        
    def generate(self, message, dataset=None, no_rag=False):
        """Generate the output message.

        Args:
            message (list[dict]): The input message.
            dataset (str, optional): The name of the dataset. Defaults to None.

        Returns:
            str: The generated message.
        """
        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'
            
        if not no_rag:
            message = self.rag(message)
            if self.is_process_image:
                return message
        
        return self.generate_inner(message, dataset)

    def chat(self, messages, dataset=None):
        """The main function for multi-turn chatting. Will call `chat_inner` with the preprocessed input messages."""
        assert hasattr(self, 'chat_inner'), 'The API model should has the `chat_inner` method. '
        for msg in messages:
            assert isinstance(msg, dict) and 'role' in msg and 'content' in msg, msg
            assert self.check_content(msg['content']) in ['str', 'dict', 'liststr', 'listdict'], msg
            msg['content'] = self.preproc_content(msg['content'])

        while len(messages):
            try:
                return self.chat_inner(messages, dataset=dataset)
            except:
                messages = messages[1:]
                while len(messages) and messages[0]['role'] != 'user':
                    messages = messages[1:]
                continue
        return 'Chat Mode: Failed with all possible conversation turns.'

    def message_to_promptimg(self, message, dataset=None):
        assert not self.INTERLEAVE
        model_name = self.__class__.__name__
        warnings.warn(
            f'Model {model_name} does not support interleaved input. '
            'Will use the first image and aggregated texts as prompt. ')
        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = None
        else:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            images = [x['value'] for x in message if x['type'] == 'image']
            if 'BLINK' == dataset:
                image = concat_images_vlmeval(images, target_size=512)
            else:
                image = images[0]
        return prompt, image

    def message_to_promptvideo(self, message):
        if self.VIDEO_LLM:
            num_videos = len([x for x in message if x['type'] == 'video'])
            if num_videos == 0:
                prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
                video = None
            else:
                prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
                video = [x['value'] for x in message if x['type'] == 'video'][0]
            return prompt, video
        else:
            import sys
            warnings.warn('Model does not support video input.')
            sys.exit(-1)
            
    def _retrieval_confidence(self, image_matrix_list, retrieval_score):
        res_retrieval_score_list = []
        for image_matrix in image_matrix_list:
            image_matrix_flatten = torch.flatten(image_matrix)
            cur_retrieval_score = retrieval_score[image_matrix_flatten].mean()
            res_retrieval_score_list.append(cur_retrieval_score)
        return res_retrieval_score_list
    
    def _answer_confidence(self, query, image_matrix, image_patch_list, chunk_size):
        image = self.map2image(image_matrix, image_patch_list, chunk_size)
        prompt = ANSWER_PROMPT.format(query)
        answer_value = self.get_confidence_value(prompt, image)
        return answer_value
        
    def rag(self, message):
        
        query = message[0]['value'] if message[0]['type'] == 'text' else message[1]['value']
        image_path = message[0]['value'] if message[0]['type'] == 'image' else message[1]['value']
        if self.mapping is not None:
            new_image_path = self.mapping.get(image_path,None)
            if new_image_path is not None and os.path.exists(new_image_path):
                new_image = Image.open(new_image_path).convert('RGB')
                if not self.is_process_image:
                    for mess in message:
                        if mess['type'] == 'image':
                            mess['value'] = new_image
                    return message
                else:
                    return new_image_path
        
        for mess in message:
            if mess['type'] == 'text':
                query = mess['value']
            elif mess['type'] == 'image':
                image_path = mess['value']
            elif mess['type'] == 'type':
                category = mess['value']
        image = Image.open(image_path).convert("RGB")
        new_image = image
        width, height = image.size
        if "llava-onevision" in self.model_path:
            self.rag_image_size = 448 # hrbench
        else:
            if max(width,height) > 5096:
                self.rag_image_size = 336
            else:
                self.rag_image_size = 224 # default 224
        
        # Create Image Memory
        embedding_image, image_matrix, image_patch_list = self.create_image_memory(image, self.rag_image_size)
        # Calculate Retrieval Score
        INSTRUCTION = "Represent this query for retrieving relevant documents: "
        embedding_query = encode([INSTRUCTION + query], self.rag_tokenizer, self.rag_model)
        retrieval_score = torch.tensor((embedding_query @ embedding_image.T)).squeeze() # observe the first image patch
        retrieval_score = 1 + retrieval_score
        retrieval_score[0] = 0.
        # Search
        answer_value = self._answer_confidence(query, image_matrix, image_patch_list, self.rag_image_size)
        retrieval_value = self._retrieval_confidence([image_matrix], retrieval_score)[0]
        confidence = answer_value
        root_node = TreeNode(image=image_matrix, confidence=confidence, depth=1, answer_score=answer_value, retrieval_score=retrieval_value)
        optimal_value = {'confidence': answer_value, 'image': image_matrix, 'answer_score': answer_value}
        open_set = [root_node]
        k_list = [0.25, 0.5, 0.75]
        num_pop = 0
        threshold_descrease = [0.1, 0.1, 0.2]
        temp_threshold_descrease = deepcopy(threshold_descrease)
        answering_confidence_threshold_upper = 1.3
        pop_num_limit = math.log((width * height) // (self.rag_image_size * self.rag_image_size), 4)
        pop_num_limit = int(pop_num_limit * 5)
        num_interval =5
        visit_leaf = False
        max_step = self.max_step
        while len(open_set) > 0 and max_step > 0:
            start_time = time.time()
            f_value = [open_set[i].confidence for i in range(len(open_set))]
            selected_index = np.argmax(f_value)
            cur_node = open_set[selected_index]
            open_set = open_set[:selected_index] + open_set[selected_index+1:]
            #if visit_leaf:
            num_pop += 1
            max_step -= 1
            
            if visit_leaf and cur_node.answer_score > answering_confidence_threshold_upper:
                break
            
            if num_pop >= pop_num_limit:
                answering_confidence_threshold_upper -= temp_threshold_descrease[0]
                if len(temp_threshold_descrease) > 1:
                    _ = temp_threshold_descrease.pop(0)
                pop_num_limit += num_interval
            
            width, height = cur_node.image.shape
            width = width * self.rag_image_size
            height = height * self.rag_image_size
            if max(width, height) <= self.rag_image_size:
                visit_leaf = True
                continue
            
            expanded_nodes = []
            sub_image_list = []
            sub_image_list = self._erosion(cur_node.image, retrieval_score, k_list, self.rag_image_size)
            retrieval_score_list = self._retrieval_confidence(sub_image_list, retrieval_score)
            assert len(retrieval_score_list) == len(sub_image_list)
            for idx in range(len(sub_image_list)):
                answer_score = self._answer_confidence(query, sub_image_list[idx], image_patch_list, self.rag_image_size)
                cur_retrieval_score = retrieval_score_list[idx].item()
                w = get_confidence_weight(cur_node.depth, self.bias_value)
                confidence = (1.-w) * cur_retrieval_score + w * answer_score
                new_node = TreeNode(depth=cur_node.depth+1, confidence=confidence, image=sub_image_list[idx], answer_score=answer_score, retrieval_score=cur_retrieval_score, parent=cur_node, select_idx=idx)
                if self.debug:
                    image_name = f'debug_image/{cur_node.depth+1}_{confidence}.png'
                    cur_sub_image = self.map2image(sub_image_list[idx],image_patch_list, self.rag_image_size)
                    cur_sub_image.save(image_name)
                    new_node.path_name = image_name
                
                expanded_nodes.append(new_node)
                if answer_score > optimal_value['answer_score']:
                    optimal_value['confidence'] = confidence
                    optimal_value['answer_score'] = answer_score
                    optimal_value['image'] = sub_image_list[idx]
                    optimal_value['node'] = new_node
        
            open_set = open_set + expanded_nodes
        
        new_image = optimal_value['image']
        new_image = self.map2image(new_image, image_patch_list, self.rag_image_size)
        if self.debug:
            new_image.save('./tmp.png')
        for mess in message:
            if mess['type'] == 'image':
                mess['value'] = new_image
        if not self.debug and self.processed_image_path is not None:
            base_name = os.path.basename(image_path)
            new_image_path = os.path.join(self.processed_image_path, base_name)
            new_image.save(new_image_path)
            if self.is_process_image:
                return new_image_path
                
        return message
        