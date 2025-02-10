from ...smp import *
import os

def report_acc_vstar(df):
    ret = defaultdict(list)
    sz = len(df['hit'])
    
    category_list = set(df['category'])
    score_dict = defaultdict(list)
    
    for i in range(sz):
        category = df['category'][i]
        gpt_score = df['hit'][i]
        score_dict[category].append(gpt_score)
        score_dict['all'].append(gpt_score)
        
    all_acc = np.mean(score_dict['all'])
    ret['type'].append('all')
    ret['acc'].append(all_acc)
    for cate in category_list:
        acc = np.mean(score_dict[cate])
        ret['type'].append(cate)
        ret['acc'].append(acc)
    
    return pd.DataFrame(ret)