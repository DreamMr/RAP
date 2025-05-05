import warnings

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *

def LOCALIZE(fname, new_fname=None):
    if new_fname is None:
        new_fname = fname.replace('.tsv', '_local.tsv')

    base_name = osp.basename(fname)
    dname = osp.splitext(base_name)[0]

    data = load(fname)
    data_new = localize_df(data, dname)
    dump(data_new, new_fname)
    print(f'The localized version of data file is {new_fname}')
    return new_fname


class ImageMCQDataset(ImageBaseDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        
    }

    DATASET_MD5 = {
    }

    
    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above.\n'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import report_acc, report_acc_MMT, mcq_circular_eval, mcq_vanilla_eval
        # assert dataset is not None
        dataset_map = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_EN_V11': 'MMBench_V11',
            'MMBench_TEST_CN': 'MMBench_CN', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11'
        }
        dataset = self.dataset_name
        if dataset in dataset_map:
            dataset = dataset_map[dataset]
        nproc = judge_kwargs.pop('nproc', 4)

        circular = False
        if listinstr(['mmbench', 'ccbench'], dataset.lower()):
            data = load(eval_file)
            data['index'] = [int(x) for x in data['index']]
            dump(data, eval_file)
            circular = True

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        if circular:
            data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        else:
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # load split
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        # May have different report acc functions for different datasets
        if 'MMT' in dataset:
            acc = report_acc_MMT(data)
        else:
            acc = report_acc(data)

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        if dataset == 'AesBench_VAL':
            warnings.warn('Note that AesBench VAL is just a toy version of AesBench TEST. For full results, \
                           please evaluate on AesBench TEST. The AesBench TEST dataset is more than 20 times \
                           larger than the VAL dataset and the leaderboard results are based on AesBench TEST.')
        return acc



class HRBenchDataset(ImageMCQDataset):

    DATASET_URL = {
        'HRBench4K': 'https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_4k.tsv',
        'HRBench8K': 'https://huggingface.co/datasets/DreamMr/HR-Bench/resolve/main/hr_bench_8k.tsv',
        'HRBench4K_single': 'hr_bench_4k_single.tsv',
        'HRBench8K_single': 'hr_bench_8k_single.tsv',
    }

    DATASET_MD5 = {
        'HRBench4K': 'f6b041b03d49543494b8a56d2e35be65',
        'HRBench8K': '274c9c7f89329b804a4723178a00219c',
        "HRBench4K_single": '6085b85e8602c943b6a850db1210f144',
        "HRBench8K_single": '8f8f41fa4f611798473e72fcac9e3c80',
    }

    def process_image(self, eval_file, processed_image_path_folder):
        assert os.path.exists(eval_file), '{} does not exist!'.format(eval_file)
        data_list = load(eval_file)
        data_dict = {}
        MAPPING_DATASET = {
            "HRBench4K_single": 'hr_bench_4k',
            "HRBench8K_single": 'hr_bench_8k',
        }
        ROOT = LMUDataRoot()
        original_image_path = os.path.join(ROOT, 'images', MAPPING_DATASET[self.dataset_name])
        #original_image_path_list = [str(x) for x in data_list['image_path']]
        processed_image_path_list = [str(x) for x in data_list['prediction']]
        index_list = [int(x) for x in data_list['index']]
        
        for idx in range(len(index_list)):
            index = index_list[idx]
            
            processed_image_path = processed_image_path_list[idx]
            #original_image_path = os.path.dirname(original_image_path_list[idx])
            for i in range(index,index+4):
                map_original_image_path = os.path.join(original_image_path,'{}.jpg'.format(i))
                data_dict[map_original_image_path] = processed_image_path
            
        
        dict_path = os.path.join(processed_image_path_folder, 'mapping.json')
        dump(data_dict, dict_path)
        return None
        

    def evaluate(self, eval_file, **judge_kwargs):
        if self.dataset_name in ['HRBench4K_single', 'HRBench8K_single']:
            processed_image_path_folder = judge_kwargs.pop('processed_image_path_folder', None)
            return self.process_image(eval_file, processed_image_path_folder)
        
        assert os.path.exists(eval_file), '{} does not exist!'.format(eval_file)
        from .utils.multiple_choice import mcq_vanilla_eval
        from .utils.hrbench import report_acc_hrbench
        nproc = judge_kwargs.pop('nproc', 4)

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'extract_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')

        if osp.exists(score_file):
            acc = load(score_file)
            return acc
        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        acc = report_acc_hrbench(data)

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        return acc
    
    
class VStarDataset(ImageMCQDataset):

    DATASET_URL = {
        'vstar': 'vstar.tsv'
    }

    DATASET_MD5 = {
        'vstar': '39ba397ba0e93d798b698db9e1421eeb'
    }
    

    def evaluate(self, eval_file, **judge_kwargs):
        assert os.path.exists(eval_file), '{} does not exist!'.format(eval_file)
        from .utils.multiple_choice import mcq_vanilla_eval
        from .utils.vstar import report_acc_vstar
        nproc = judge_kwargs.pop('nproc', 4)

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'extract_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')

        if osp.exists(score_file):
            acc = load(score_file)
            return acc
        
        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)
        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        acc = report_acc_vstar(data)

        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        return acc
    


class CustomMCQDataset(ImageMCQDataset):

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)
