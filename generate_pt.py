import argparse
import json

import torch
from transformers import *
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from modeling.generator import Generator

torch.set_num_threads(4)

# for REPRODUCIBILITY
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--cp_pair_path', type=str, default='/home/jun/project/MHGRN/data/obqa/hybrid/dev_cpt_pairs_1hop_hybrid.jsonl')
    parser.add_argument("--start_idx", default=-1, type=int)
    parser.add_argument("--end_idx", default=-1, type=int)

    # model
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--output_len', type=int, default=31)
    parser.add_argument('--context_len', type=int, default=16)

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ----------------------------------------------------- #

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_tokens(['<PAD>'])
    tokenizer.add_tokens(['<SEP>'])
    tokenizer.add_tokens(['<END>'])
    PAD = tokenizer.convert_tokens_to_ids('<PAD>')
    SEP = tokenizer.convert_tokens_to_ids('<SEP>')
    END = tokenizer.convert_tokens_to_ids('<END>')
    all_evidence_prompt = []
    all_evidence_num = []
    with open(args.cp_pair_path, 'r') as f:
        data_lst = [json.loads(line) for line in f]
        if args.start_idx == -1:
            args.start_idx = 0
        if args.end_idx == -1:
            args.end_idx = len(data_lst)
        data_lst = data_lst[args.start_idx: args.end_idx]

        for data in tqdm(data_lst):
            cp_pair_lst = data['non_adj_cp_pair']
            all_evidence_num.append(len(cp_pair_lst))
            for subj, obj in cp_pair_lst:
                subj = subj.replace('_', ' ')
                obj = obj.replace('_', ' ')
                context = obj + '<SEP>' + subj
                context = tokenizer.encode(context, add_special_tokens=False)[:args.context_len]
                context += [PAD] * (args.context_len - len(context))
                all_evidence_prompt.append(context)
    prompt_dataset = TensorDataset(torch.tensor(all_evidence_prompt, dtype=torch.long))

    # self define lm head gpt2
    config = GPT2Config.from_pretrained('gpt2')
    config.vocab_size = len(tokenizer)
    gpt = GPT2Model.from_pretrained('gpt2')
    gpt.resize_token_embeddings(len(tokenizer))
    generator = Generator(gpt, config, max_len=args.output_len).to(args.device)
    generator.load_state_dict(torch.load(
        '/home/jun/project/Commonsense-Path-Generator/generator_ckpt/commonsense-path-generator_transformers_2.8.0.ckpt',
        map_location=args.device))
    generator.eval()

    data_sampler = SequentialSampler(prompt_dataset)
    dataloader = DataLoader(prompt_dataset, sampler=data_sampler, batch_size=args.batch_size)
    feature_tensor = torch.zeros(len(all_evidence_prompt), gpt.config.hidden_size)
    start_idx = 0

    for i, context in enumerate(tqdm(dataloader, desc="Path Generation")):
        context = context[0].to(args.device)
        end_idx = start_idx + context.size(0)

        with torch.no_grad():
            context_embedding, generated_paths = generator(context, train=False, return_path=True)
        feature_tensor[start_idx: end_idx] = context_embedding
        start_idx = end_idx

        if i % 1000 == 0:
            for path in generated_paths:
                path = tokenizer.decode(path.tolist(), skip_special_tokens=True)
                path = ' '.join(path.replace('<PAD>', '').split())
                print(path)
    assert start_idx == len(all_evidence_prompt)
    output_dic = {'all_evidence_vecs': feature_tensor, 'all_evidence_num': all_evidence_num}
    save_name = f'{args.cp_pair_path}.pt_{args.start_idx}_{args.end_idx}'
    print(f'Saving to {save_name}...')
    torch.save(output_dic, save_name)


if __name__ == '__main__':
    main()