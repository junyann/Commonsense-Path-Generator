import os

import torch

folder = '/home/jun/project/MHGRN/data/obqa/hybrid'
pt_list = [
    'train_cpt_pairs_1hop_hybrid.jsonl.pt_0_4000',
    'train_cpt_pairs_1hop_hybrid.jsonl.pt_4000_8000',
    'train_cpt_pairs_1hop_hybrid.jsonl.pt_8000_12000',
    'train_cpt_pairs_1hop_hybrid.jsonl.pt_12000_16000',
    'train_cpt_pairs_1hop_hybrid.jsonl.pt_16000_20000'
]
save_name = 'train_cpt_pairs_1hop_hybrid.jsonl.pt'

output_dic = {'all_evidence_vecs': [], 'all_evidence_num': []}
tensor_lst = []
for pt_name in pt_list:
    pt_data = torch.load(os.path.join(folder, pt_name))
    output_dic['all_evidence_num'] += pt_data['all_evidence_num']
    output_dic['all_evidence_vecs'].append(pt_data['all_evidence_vecs'])
output_dic['all_evidence_vecs'] = torch.cat(output_dic['all_evidence_vecs'], 0)
torch.save(output_dic, os.path.join(folder, save_name))
