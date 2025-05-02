from rap.smp import *
import copy
file_list = [
    # r'/mnt/data/users/wenbinwang/datasets/LVLM_Benchmark/LMUData/hr_bench_4k_single.tsv',
    # r'/mnt/data/users/wenbinwang/datasets/LVLM_Benchmark/LMUData/hr_bench_8k_single.tsv'
    r'/mnt/data/users/wenbinwang/datasets/LVLM_Benchmark/LMUData/vstar.tsv'
]
save_root = r'/mnt/data/users/wenbinwang/datasets/LVLM_Benchmark'
md5_dic = {}
for file_path in file_list:
    dataset = load(file_path)
    new_dataset = copy.deepcopy(dataset)

    #new_dataset['image'] = dataset['image_path'].apply(lambda x: encode_image_file_to_base64(x))
    #new_dataset.drop('image_path', axis=1, inplace=True)

    file_name = file_path.split('/')[-1]
    save_path = os.path.join(save_root, file_name)
    dump(new_dataset, save_path)

    if os.path.exists(save_path):
        cur_md5 = md5(save_path)
        md5_dic[save_path] = cur_md5
    
print(md5_dic)