import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model


# Define the generator model
class Generator(nn.Module):
    def __init__(self, gpt, config, max_len=31):
        super(Generator, self).__init__()
        self.gpt = gpt
        self.config = config
        self.max_len = max_len
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, inputs):
        # input: [batch, seq]
        context_len = inputs.size(1)
        generated = inputs
        next_token = inputs
        past = None
        with torch.no_grad():
            for step in range(self.max_len):
                outputs = self.gpt(next_token, past=past)
                hidden = outputs[0][:, -1]
                past = outputs[1]
                next_token_logits = self.lm_head(hidden)
                next_logits, next_token = next_token_logits.topk(k=1, dim=1)
                generated = torch.cat((generated, next_token), dim=1)
        return generated


lm_type = 'gpt2'
config = GPT2Config.from_pretrained(lm_type)
tokenizer = GPT2Tokenizer.from_pretrained(lm_type)
tokenizer.add_tokens(['<PAD>'])
tokenizer.add_tokens(['<SEP>'])
tokenizer.add_tokens(['<END>'])
gpt = GPT2Model.from_pretrained(lm_type)
config.vocab_size = len(tokenizer)
gpt.resize_token_embeddings(len(tokenizer))
pretrain_generator_ckpt = "/home/jun/project/Commonsense-Path-Generator/generator_ckpt/commonsense-path-generator_transformers_2.8.0.ckpt"
generator = Generator(gpt, config)
generator.load_state_dict(torch.load(pretrain_generator_ckpt, map_location='cpu'))


def prepare_input(head_entity, tail_entity, input_len=16):
    head_entity = head_entity.replace('_', ' ')
    tail_entity = tail_entity.replace('_', ' ')
    input_token = tail_entity + '<SEP>' + head_entity
    input_id = tokenizer.encode(input_token, add_special_tokens=False)[:input_len]
    input_id += [tokenizer.convert_tokens_to_ids('<PAD>')] * (input_len - len(input_id))
    return torch.tensor([input_id], dtype=torch.long)


def connect_entities(head_entity, tail_entity):
    gen_input = prepare_input(head_entity, tail_entity)
    gen_output = generator(gen_input)
    path = tokenizer.decode(gen_output[0].tolist(), skip_special_tokens=True)
    path = ' '.join(path.replace('<PAD>', '').split())
    return path[path.index('<SEP>')+6:]


head_entity = 'curiosity'
tail_entity = 'hear_news'
path = connect_entities(head_entity, tail_entity)
print(path)
