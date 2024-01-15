import torch
from typing import Tuple
import math
from tqdm.auto import tqdm, trange
import statistics
import time

def cache_size(kv_cache):
    """
    the amount of memory(MB)
    """
    cnt = 0
    for i in range(len(kv_cache)):
        for tensor in kv_cache[i]:
            cnt += tensor.numel()
    return cnt*2/(1024**2)

def gpu_stats():
    torch.cuda.empty_cache()
    memory_stats = torch.cuda.memory_stats()
    print("Current GPU memory usage:", round(memory_stats["allocated_bytes.all.current"]/(1024**3), 3), "GB")
    print("Peak GPU memory usage:", round(memory_stats["allocated_bytes.all.peak"]/(1024**3), 3), "GB")
    print("Reserved GPU memory:", round(memory_stats["reserved_bytes.all.allocated"]/(1024**3), 3), "GB")


# ANSI code for different colors
class Color:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'

    @staticmethod
    def print(content, color: str):
        print(f"{getattr(Color, color.upper())}{content}{Color.RESET}")


def relu_normalize(p, q):
    """
    Construct the modified sampling distribution
    """
    tmp_dist = torch.relu(p-q)
    return tmp_dist / tmp_dist.sum(dim=-1, keepdim=True)

def entropy(p):
    """
    Shano entropy of a distribution
    """
    return -torch.sum(p*p.log(), dim=-1)

def truncate_kv_cache_silo(kv_cache, eviction_ids):
    kv_cache = list(kv_cache)
    for i in range(len(kv_cache)):
        kv_cache[i] = list(kv_cache[i])
    l = kv_cache[0][0].shape[2]
    head_dim = kv_cache[0][0].shape[-1]
    num_heads = kv_cache[0][0].shape[1]
    for i in range(len(eviction_ids)):
        _index = torch.arange(l, device=kv_cache[0][0].device).unsqueeze(0).repeat(len(eviction_ids[i]), 1) # (num_heads, l)
        mask = (_index != torch.tensor(eviction_ids[i], device=kv_cache[0][0].device).unsqueeze(-1)) # (num_heads, l)
        kv_cache[i][0] = kv_cache[i][0][0][mask, ...].view(1, num_heads, -1, head_dim)
        kv_cache[i][1] = kv_cache[i][1][0][mask, ...].view(1, num_heads, -1, head_dim)
    return kv_cache

def truncate_kv_cache_liso(kv_cache, eviction_ids):
    kv_cache = list(kv_cache)
    for i in range(len(kv_cache)):
        kv_cache[i] = list(kv_cache[i])
    l = kv_cache[0][0].shape[2]
    head_dim = kv_cache[0][0].shape[-1]
    num_heads = kv_cache[0][0].shape[1]
    for i in range(eviction_ids.shape[0]):
        src_ = torch.zeros(num_heads, eviction_ids.shape[-1]).to(kv_cache[0][0].device)
        mask = torch.ones(num_heads, l, device=kv_cache[0][0].device).scatter(dim=-1, index=eviction_ids[i], src=src_).bool()
        kv_cache[i][0] = kv_cache[i][0][0][mask, ...].view(1, num_heads, -1, head_dim)
        kv_cache[i][1] = kv_cache[i][1][0][mask, ...].view(1, num_heads, -1, head_dim)
    return kv_cache


def truncate_kv_cache(kv_cache: Tuple, start, end):
    remain_id = torch.tensor(list(sorted(list(set(list(range(kv_cache[0][0].shape[2]))).difference(set(list(range(start, end))))))), device=kv_cache[0][0].device)
    kv_cache = list(kv_cache)
    for i in range(len(kv_cache)):
        kv_cache[i] = list(kv_cache[i])
        kv_cache[i][0] = kv_cache[i][0][:, :, remain_id, :]
        kv_cache[i][1] = kv_cache[i][1][:, :, remain_id, :]
    return kv_cache


def logits_adapter(logits: torch.Tensor, temperature: float, top_p: float):
    """
    Apply given transformation to the input logits, including temperature scaling and top_p renormalization
    """
    flag = False
    if logits.ndim==3:
        bsz = logits.shape[0]
        l = logits.shape[1]
        logits = logits.view(-1, logits.shape[-1])
        flag = True
    prob = torch.softmax(logits / temperature, dim=-1)
    sorted_prob, sorted_prob_idx = torch.sort(prob, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_prob, dim=-1)
    mask = (cumsum - sorted_prob) > top_p
    sorted_prob[mask] = 0.0
    sorted_prob.div_(sorted_prob.sum(dim=-1, keepdim=True))
    _, gather_pos = torch.sort(sorted_prob_idx, descending=False, dim=-1)
    final_prob = torch.gather(sorted_prob, -1, gather_pos)
    if flag: final_prob = final_prob.view(bsz, l, -1)
    return final_prob, torch.softmax(logits, dim=-1)


def h2o_head_decay_score(attention_map, decay_factor, device, stride):
    num_heads = attention_map[0].shape[1]
    num_layers = len(attention_map)
    budget = attention_map[0].shape[-1]
    cache_attn_scores = torch.tensor([[[0.0]*(budget+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=device)
    decay_tensor = torch.tensor([decay_factor**power for power in range(budget)], device=device).flip(dims=(0,)).unsqueeze(-1).unsqueeze(0).repeat(num_heads, 1, budget) # (1, budget, budget)
    for l in range(num_layers):
        cache_attn_scores[l, :, :-stride] = torch.sum(attention_map[l][0] * decay_tensor, dim=1) * (1.0 - decay_factor)
    return cache_attn_scores

def h2o_head_decay_prob_score(attention_map, decay_factor, device, probs):
    num_heads = attention_map[0].shape[1]
    num_layers = len(attention_map)
    budget = attention_map[0].shape[-1]
    cache_attn_scores = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=device)
    decay_tensor = torch.tensor([decay_factor**power for power in range(budget)], device=device).flip(dims=(0,)).unsqueeze(-1).unsqueeze(0).repeat(num_heads, 1, budget) # (1, budget, budget)
    probs = torch.tensor(probs, device=device).unsqueeze(-1).unsqueeze(0).repeat(num_heads, 1, budget) # (num_heads, budgett, budget)
    for l in range(num_layers):
        cache_attn_scores[l, :, :-1] = torch.sum(attention_map[l][0] * decay_tensor * probs, dim=1) * (1.0 - decay_factor)
    return cache_attn_scores

def h2o_head_prob_score(attention_map, device, probs, mode:str='v1'):
    num_heads = attention_map[0].shape[1]
    num_layers = len(attention_map)
    budget = attention_map[0].shape[-1]
    cache_attn_scores = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=device)
    cache_attn_scores_square = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=device)
    if mode == 'v1':
        probs = 1.0-torch.tensor(probs, device=device).unsqueeze(-1).unsqueeze(0).repeat(num_heads, 1, budget) # (num_heads, budgett, budget)
    elif mode == 'v2':
        probs = torch.tensor(probs, device=device).unsqueeze(-1).unsqueeze(0).repeat(num_heads, 1, budget) # (num_heads, budgett, budget)
    for l in range(num_layers):
        cache_attn_scores[l, :, :-1] = torch.sum(attention_map[l][0] * probs, dim=1)
        cache_attn_scores_square[l, :, :-1] = torch.sum((attention_map[l][0] * probs)**2, dim=1)
    return cache_attn_scores, cache_attn_scores_square

def h2o_head_score(attention_map, device, stride, num_heads):
    num_layers = len(attention_map)
    budget = attention_map[0].shape[-1]
    cache_attn_scores = torch.tensor([[[0.0]*(budget+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=device)
    cache_attn_scores_square = torch.tensor([[[0.0]*(budget+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=device)
    for l in range(num_layers):
        cache_attn_scores[l, :, :-stride] = torch.sum(attention_map[l][0], dim=1)
        cache_attn_scores_square[l, :, :-stride] = torch.sum(attention_map[l][0]**2, dim=1)
    return cache_attn_scores, cache_attn_scores_square


@torch.inference_mode()
def generate(self, input_ids, generation_config, kv_mode='encoding', stride=1):
    """
    generation utility function suppr=orting:
    (1). long input, short output
    (2). short input, long output
    mode in [silo, liso]
    """
    temperature = generation_config['temperature']
    top_p = generation_config['top_p']
    max_new_tokens = generation_config['max_new_tokens']
    budget = generation_config['budget']
    mode = generation_config['kv_policy']
    temp_length = generation_config.get('temp_length', 4)
    recent_ratio = generation_config.get('recent_ratio', 0.1)
    num_layers = self.config.num_hidden_layers
    if not hasattr(self.config, "num_key_value_heads"): num_heads = self.config.num_attention_heads
    else: num_heads = self.config.num_key_value_heads
    tokenizer = self.tokenizer
    # Handle MQA and GQA
    is_gqa = hasattr(self.config, "num_key_value_heads") and getattr(self.config, "num_key_value_heads") != getattr(self.config, "num_attention_heads")
    if is_gqa: rep_n = self.config.num_attention_heads // self.config.num_key_value_heads
    else: rep_n = 1
    if kv_mode == 'decoding':
        """
        auto-regressive decoding
        """
        outputs_prefilling = self(input_ids=input_ids, use_cache=True)
        prefix_token_lst = input_ids[0].cpu().numpy().tolist()
        past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
        logits_prev_step = logits[:, -1, :]
        prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)
        decay_factor = math.exp(math.log(0.001) / budget)

        cache_tokens = []
        cache_probs = []
        cache_cur_probs = []
        cache_positions = []
        cache_attn_scores = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
        cache_attn_scores_decay_avg_std = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
        cache_attn_scores_square = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
        cache_counter = torch.tensor([[[1.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
        cache_counter = torch.cumsum(cache_counter, dim=-1).flip(dims=(2,)) - 1.0
        cache_counter_token = torch.tensor([1.0]*(budget+1), device=self.device)
        cache_counter_token = torch.cumsum(cache_counter_token, dim=-1).flip(dims=(0, )) - 1.0
        n = 0
        output_ids = []
        token_probs = []
        cur_pos_id = past_key_values[0][0].shape[2]
        evicted_positions = []
        while n < max_new_tokens:
            next_token = torch.multinomial(prob_prev_step, num_samples=1)
            output_ids.append(next_token[0, 0].cpu().item())
            next_token_prob = torch.gather(raw_prob_prev_step, -1, next_token) # (bsz, 1)
            token_probs.append((tokenizer.convert_ids_to_tokens([output_ids[-1]])[0], next_token_prob[0, 0].cpu().item()))
            n += 1
            if output_ids[-1] == tokenizer.eos_token_id: break
            outputs = self(input_ids=next_token, 
                            past_key_values=past_key_values,
                            attention_mask=torch.ones(next_token.shape[0], 1+past_key_values[0][0].shape[2], dtype=torch.long, device=next_token.device),
                            position_ids=torch.LongTensor([cur_pos_id]).to(self.device).view(-1, 1),
                            use_cache=True,
                            output_attentions=True)
            # unified processing for GQA and MHA
            outputs.attentions = list(outputs.attentions)
            for l in range(num_layers):
                bs = outputs.attentions[l].shape[0]
                sl = outputs.attentions[l].shape[2]
                tl = outputs.attentions[l].shape[3]
                outputs.attentions[l] = outputs.attentions[l].reshape(bs, num_heads, rep_n, sl, tl).mean(dim=2) # (bs, num_kv_heads, sl, tl)
            past_key_values = outputs.past_key_values
            logits_prev_step = outputs.logits[:, -1, :]
            cache_cur_probs.append(torch.exp(-entropy(raw_prob_prev_step))[0].cpu().item())
            prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)

            # update 
            cache_probs.append(next_token_prob[0,0].cpu().item())
            cache_tokens.append(output_ids[-1])
            cache_positions.append(cur_pos_id)

            # update accumulated attention scores
            if 'h2o_head' == mode or 'h2o_head_avg' == mode:
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                    cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
            elif 'h2o_head_std' == mode or 'h2o_head_std_avg' == mode:
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                    cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                    cache_attn_scores_square[l, :, :attention_map.shape[-1]] += attention_map ** 2
            elif 'h2o_head_decay' == mode or 'h2o_head_decay_avg' == mode:
                a = 0.96
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                    cache_attn_scores[l, :, :attention_map.shape[-1]] = a * cache_attn_scores[l, :, :attention_map.shape[-1]] + (1-a) * attention_map
            elif 'h2o_head_decay_avg_std' == mode:
                a = 0.96
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                    cache_attn_scores[l, :, :attention_map.shape[-1]] = a * cache_attn_scores[l, :, :attention_map.shape[-1]] + (1-a) * attention_map
                    cache_attn_scores_decay_avg_std[l, :, :attention_map.shape[-1]] += cache_attn_scores[l, :, :attention_map.shape[-1]]
                    cache_attn_scores_square[l, :, :attention_map.shape[-1]] += (cache_attn_scores[l, :, :attention_map.shape[-1]])**2
            elif 'tova' == mode:
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                    cache_attn_scores[l, :, :attention_map.shape[-1]] = attention_map
            # evict if current kv cache size exceeds the budget
            cur_kv_size = past_key_values[0][0].shape[2]
            if (cur_kv_size-len(prefix_token_lst)) > budget and mode != 'full':
                cache_counter += 1.0
                cache_counter_token += 1.0
                probs_tensor = torch.tensor(cache_probs, device=self.device)
                positions_tensor = torch.tensor(cache_positions, device=self.device).float()
                positions_tensor = positions_tensor / float(cur_pos_id)
                recent_ratio = 0.3
                recent_window = int(budget*recent_ratio)
                if mode == 'probability':
                    scores = probs_tensor
                    scores[-recent_window:] = -1e9
                    _, evict_id = torch.topk(scores, k=1, dim=-1)
                    evict_id = evict_id[0].cpu().item()
                    past_key_values = truncate_kv_cache(past_key_values, start=len(prefix_token_lst)+evict_id, end=len(prefix_token_lst)+evict_id+1)
                    evicted_positions.append(cache_positions[evict_id]-len(prefix_token_lst))
                    cache_probs.pop(evict_id)
                    cache_tokens.pop(evict_id)
                    cache_cur_probs.pop(evict_id)
                    cache_positions.pop(evict_id)
                elif mode in ['h2o_head', 'h2o_head_decay', 'h2o_head_avg', 'h2o_head_decay_avg', 'h2o_head_prob', 'h2o_head_prob_avg', 'h2o_head_probv2', 'h2o_head_probv2_avg', 'h2o_head_decay_prob', 'h2o_head_decay_probv2', 'h2o_head_decay_prob_avg', 'h2o_head_decay_probv2_avg']:
                    if not 'avg' in mode:
                        eviction_ids = torch.argmin(cache_attn_scores[:, :, :-recent_window], dim=-1) + len(prefix_token_lst)
                    else:
                        eviction_ids = torch.argmin(cache_attn_scores[:, :, :-recent_window] / cache_counter[:, :, :-recent_window], dim=-1) + len(prefix_token_lst)
                    _eviction_ids = eviction_ids
                    eviction_ids = eviction_ids.cpu().numpy().tolist()
                    past_key_values = truncate_kv_cache_silo(past_key_values, eviction_ids)
                    _index = torch.arange(cache_attn_scores.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).repeat(num_layers, num_heads, 1)
                    _eviction_ids -= len(prefix_token_lst)
                    mask = (_eviction_ids.unsqueeze(-1)!=_index).view(-1, _index.shape[-1])
                    cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, _index.shape[-1]-1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                    if 'avg' in mode:
                        cache_counter = torch.cat((cache_counter.view(-1, cache_counter.shape[-1])[mask].view(num_layers, num_heads, _index.shape[-1]-1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                elif mode in ['h2o_head_std', 'h2o_head_std_avg']:
                    cur_std = torch.sqrt(cache_attn_scores_square / cache_counter - (cache_attn_scores / cache_counter)**2)
                    cur_std[:, :, -10:] = 1e9
                    _, feasible_ids = torch.topk(cur_std, largest=False, k=budget-recent_window, dim=-1) # (layers, heads, k)
                    if 'avg' in mode:
                        argmin_id = torch.argmin(cache_attn_scores.gather(dim=-1, index=feasible_ids) / cache_counter.gather(dim=-1, index=feasible_ids), dim=-1).unsqueeze(-1) # (layers, heads)
                    else:
                        argmin_id = torch.argmin(cache_attn_scores.gather(dim=-1, index=feasible_ids), dim=-1).unsqueeze(-1) # (layers, heads)
                    eviction_ids = feasible_ids.gather(dim=-1, index=argmin_id).squeeze(-1) + len(prefix_token_lst)
                    _eviction_ids = eviction_ids
                    eviction_ids = eviction_ids.cpu().numpy().tolist()
                    past_key_values = truncate_kv_cache_silo(past_key_values, eviction_ids)
                    _index = torch.arange(cache_attn_scores.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).repeat(num_layers, num_heads, 1)
                    _eviction_ids -= len(prefix_token_lst)
                    mask = (_eviction_ids.unsqueeze(-1)!=_index).view(-1, _index.shape[-1])
                    cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, _index.shape[-1]-1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                    cache_attn_scores_square = torch.cat((cache_attn_scores_square.view(-1, cache_attn_scores_square.shape[-1])[mask].view(num_layers, num_heads, _index.shape[-1]-1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                    if 'avg' in mode:
                        cache_counter = torch.cat((cache_counter.view(-1, cache_counter.shape[-1])[mask].view(num_layers, num_heads, _index.shape[-1]-1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                elif mode in ['h2o_head_decay_avg_std']:
                    cur_std = torch.sqrt(cache_attn_scores_square / cache_counter - (cache_attn_scores_decay_avg_std / cache_counter)**2)
                    cur_std[:, :, -10:] = 1e9
                    _, feasible_ids = torch.topk(cur_std, largest=False, k=budget-recent_window, dim=-1) # (layers, heads, k)
                    argmin_id = torch.argmin(cache_attn_scores.gather(dim=-1, index=feasible_ids) / cache_counter.gather(dim=-1, index=feasible_ids), dim=-1).unsqueeze(-1) # (layers, heads)
                    eviction_ids = feasible_ids.gather(dim=-1, index=argmin_id).squeeze(-1) + len(prefix_token_lst)
                    _eviction_ids = eviction_ids
                    eviction_ids = eviction_ids.cpu().numpy().tolist()
                    past_key_values = truncate_kv_cache_silo(past_key_values, eviction_ids)
                    _index = torch.arange(cache_attn_scores.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).repeat(num_layers, num_heads, 1)
                    _eviction_ids -= len(prefix_token_lst)
                    mask = (_eviction_ids.unsqueeze(-1)!=_index).view(-1, _index.shape[-1])
                    cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, _index.shape[-1]-1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                    cache_attn_scores_decay_avg_std = torch.cat((cache_attn_scores_decay_avg_std.view(-1, cache_attn_scores_decay_avg_std.shape[-1])[mask].view(num_layers, num_heads, _index.shape[-1]-1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                    cache_attn_scores_square = torch.cat((cache_attn_scores_square.view(-1, cache_attn_scores_square.shape[-1])[mask].view(num_layers, num_heads, _index.shape[-1]-1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                    cache_counter = torch.cat((cache_counter.view(-1, cache_counter.shape[-1])[mask].view(num_layers, num_heads, _index.shape[-1]-1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                elif mode == 'tova':
                    eviction_ids = torch.argmin(cache_attn_scores, dim=-1) + len(prefix_token_lst)
                    _eviction_ids = eviction_ids
                    eviction_ids = eviction_ids.cpu().numpy().tolist()
                    past_key_values = truncate_kv_cache_silo(past_key_values, eviction_ids)
                    _index = torch.arange(cache_attn_scores.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).repeat(num_layers, num_heads, 1)
                    _eviction_ids -= len(prefix_token_lst)
                    mask = (_eviction_ids.unsqueeze(-1)!=_index).view(-1, _index.shape[-1])
                    cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, _index.shape[-1]-1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                elif mode == 'recency':
                    scores = 1.0 - positions_tensor
                    _, evict_id = torch.topk(scores, k=1, dim=-1)
                    evict_id = evict_id[0].cpu().item()
                    past_key_values = truncate_kv_cache(past_key_values, start=len(prefix_token_lst)+evict_id, end=len(prefix_token_lst)+evict_id+1)
                    evicted_positions.append(cache_positions[evict_id]-len(prefix_token_lst))
                    cache_probs.pop(evict_id)
                    cache_tokens.pop(evict_id)
                    cache_cur_probs.pop(evict_id)
                    cache_positions.pop(evict_id)
                elif mode == 'random':
                    scores = torch.rand(*positions_tensor.shape).to(self.device)
                    _, evict_id = torch.topk(scores, k=1, dim=-1)
                    evict_id = evict_id[0].cpu().item()
                    past_key_values = truncate_kv_cache(past_key_values, start=len(prefix_token_lst)+evict_id, end=len(prefix_token_lst)+evict_id+1)
                    evicted_positions.append(cache_positions[evict_id]-len(prefix_token_lst))
                    cache_probs.pop(evict_id)
                    cache_tokens.pop(evict_id)
                    cache_cur_probs.pop(evict_id)
                    cache_positions.pop(evict_id)
            cur_pos_id += 1
        _tmp = past_key_values[0][0].shape[2]-len(prefix_token_lst)
        print(f"KV cache budget ratio: {_tmp / len(output_ids) *100:.2f}%({_tmp}/{len(output_ids)})")
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    elif kv_mode == 'encoding':
        """
        prompt encoding/prefilling
        """
        length = input_ids.shape[-1]
        if budget >= 1.0:
            outputs_prefilling = self(input_ids=input_ids, use_cache=True)
            past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
            logits_prev_step = logits[:, -1, :]
            prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)
            cur_pos_id = past_key_values[0][0].shape[2]
        else:
            s = time.time()
            budget = int(length * budget) + stride
            for idx in range(budget, -1, -1):
                if (length-idx)%stride==0: break
            prefix = input_ids[:, :idx]
            recent_window = int(budget*recent_ratio)
            decay_factor = math.exp(math.log(0.001) / budget)
            sink_length = temp_length
            outputs_prefilling = self(input_ids=prefix, use_cache=True, output_attentions=True)
            past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits[:, :-1].view(-1, logits.shape[-1]), prefix[:, 1:].clone().view(-1))
            prefix_prob = torch.exp(-loss).cpu().numpy().tolist()
            logits_prev_step = logits[:, -1, :]
            _, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)
            prefix_token_lst = input_ids[0].cpu().numpy().tolist()
            cache_tokens = prefix[0].cpu().numpy().tolist()
            outputs_prefilling.attentions = list(outputs_prefilling.attentions)
            for l in range(num_layers):
                bs = outputs_prefilling.attentions[l].shape[0]
                sl = outputs_prefilling.attentions[l].shape[2]
                tl = outputs_prefilling.attentions[l].shape[3]
                outputs_prefilling.attentions[l] = outputs_prefilling.attentions[l].reshape(bs, num_heads, rep_n, sl, tl).mean(dim=2) # (bs, num_kv_heads, sl, tl)
            if 'decay' in mode and not 'prob' in mode:
                cache_attn_scores = h2o_head_decay_score(outputs_prefilling.attentions, decay_factor, self.device)
            else:
                cache_attn_scores, cache_attn_scores_square = h2o_head_score(outputs_prefilling.attentions, self.device, stride, num_heads)
            del outputs_prefilling
            torch.cuda.empty_cache()
            cache_counter = torch.tensor([[[1.0]*(idx+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
            cache_counter = torch.cumsum(cache_counter, dim=-1).flip(dims=(2,)) - float(stride)
            cache_counter_token = torch.tensor([1.0]*(idx+stride), device=self.device)
            cache_counter_token = torch.cumsum(cache_counter_token, dim=-1).flip(dims=(0, )) - float(stride)
            n = 0
            output_ids = []
            token_probs = []
            cur_pos_id = past_key_values[0][0].shape[2]
            evicted_positions = []
            log_probs = []
            for token_i in range(idx, length, stride):
                n += stride
                outputs = self(input_ids=input_ids[:, token_i:token_i+stride],
                                past_key_values=past_key_values,
                                attention_mask=torch.ones(1, stride+past_key_values[0][0].shape[2], dtype=torch.long, device=self.device),
                                position_ids=torch.LongTensor(list(range(cur_pos_id, cur_pos_id+stride))).to(self.device).view(1, -1),
                                use_cache=True,
                                output_attentions=True)
                past_key_values = outputs.past_key_values
                logits_prev_step = outputs.logits[:, -1, :]
                prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)

                # Unified processing for MQA, GQA and MHA
                outputs.attentions = list(outputs.attentions)
                for l in range(num_layers):
                    bs = outputs.attentions[l].shape[0]
                    sl = outputs.attentions[l].shape[2]
                    tl = outputs.attentions[l].shape[3]
                    outputs.attentions[l] = outputs.attentions[l].reshape(bs, num_heads, rep_n, sl, tl).mean(dim=2) # (bs, num_kv_heads, sl, tl)

                # update accumulated attention scores
                if 'h2o_head' == mode or 'h2o_head_avg' == mode:
                    for l in range(num_layers):
                        attention_map = outputs.attentions[l][0, :, :, :].sum(dim=1) # (num_heads, stride, stride+l)
                        cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                elif 'h2o_head_std' == mode or 'h2o_head_std_avg' == mode:
                    for l in range(num_layers):
                        attention_map = outputs.attentions[l][0, :, :, :].sum(dim=1) # (num_heads, l)
                        attention_map_sq = ((outputs.attentions[l][0, :, :, :])**2).sum(dim=1)
                        cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                        cache_attn_scores_square[l, :, :attention_map.shape[-1]] += attention_map_sq
                elif 'h2o_head_decay' == mode or 'h2o_head_decay_avg' == mode:
                    for l in range(num_layers):
                        a = decay_factor
                        attention_map = outputs.attentions[l][0, :, 0, :] # (num_heads, l)
                        cache_attn_scores[l, :, :attention_map.shape[-1]] = a * cache_attn_scores[l, :, :attention_map.shape[-1]] + (1-a) * attention_map
                elif 'tova' == mode:
                    for l in range(num_layers):
                        attention_map = outputs.attentions[l][0, :, -1, :] # (num_heads, l)
                        cache_attn_scores[l, :, :attention_map.shape[-1]] = attention_map
                # evict if current kv cache size exceeds the budget
                cur_kv_size = past_key_values[0][0].shape[2]
                if mode != 'full':
                    cache_counter += float(stride)
                    cache_counter_token += float(stride)
                    if mode in ['h2o_head', 'h2o_head_decay', 'h2o_head_avg', 'h2o_head_decay_avg']:
                        if not 'avg' in mode:
                            eviction_ids = torch.topk(cache_attn_scores[:, :, sink_length:-recent_window], dim=-1, k=stride, largest=False)[1] + sink_length
                        else:
                            eviction_ids = torch.topk(cache_attn_scores[:, :, sink_length:-recent_window] / cache_counter[:, :, sink_length:-recent_window], dim=-1, k=stride, largest=False)[1] + sink_length
                        past_key_values = truncate_kv_cache_liso(past_key_values, eviction_ids)
                        _index = torch.ones(num_layers, num_heads, cache_attn_scores.shape[-1], device=self.device).view(num_layers*num_heads, -1)
                        _src = torch.zeros(num_layers, num_heads, stride, device=self.device).view(num_layers*num_heads, -1)
                        mask = _index.scatter(dim=-1, index=eviction_ids.view(num_layers*num_heads, -1), src=_src).bool()
                        cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, stride, device=self.device)), dim=-1)
                        cache_counter = torch.cat((cache_counter.view(-1, cache_counter.shape[-1])[mask].view(num_layers, num_heads, -1), (torch.arange(stride)-stride+1).view(1, 1, -1).repeat(num_layers, num_heads, 1).flip(dims=(2,)).to(self.device)), dim=-1)
                    elif mode in ['h2o_head_std', 'h2o_head_std_avg']:
                        cur_std = torch.sqrt(cache_attn_scores_square / cache_counter - (cache_attn_scores / cache_counter)**2)
                        cur_std[:, :, -10:] = 1e9
                        cur_std[:, :, :sink_length] = 1e9
                        _, feasible_ids = torch.topk(cur_std, largest=False, k=max(budget-recent_window-sink_length, stride), dim=-1) # (layers, heads, k)
                        if 'avg' in mode:
                            argmin_id = torch.topk(cache_attn_scores.gather(dim=-1, index=feasible_ids) / cache_counter.gather(dim=-1, index=feasible_ids), dim=-1, largest=False, k=stride)[1] # (layers, heads)
                        else:
                            argmin_id = torch.topk(cache_attn_scores.gather(dim=-1, index=feasible_ids), dim=-1, largest=False, k=stride)[1] # (layers, heads)
                        eviction_ids = feasible_ids.gather(dim=-1, index=argmin_id)
                        past_key_values = truncate_kv_cache_liso(past_key_values, eviction_ids)
                        _index = torch.ones(num_layers, num_heads, cache_attn_scores.shape[-1], device=self.device).view(num_layers*num_heads, -1)
                        _src = torch.zeros(num_layers, num_heads, stride, device=self.device).view(num_layers*num_heads, -1)
                        mask = _index.scatter(dim=-1, index=eviction_ids.view(num_layers*num_heads, -1), src=_src).bool()
                        cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, stride, device=self.device)), dim=-1)
                        cache_attn_scores_square = torch.cat((cache_attn_scores_square.view(-1, cache_attn_scores_square.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, stride, device=self.device)), dim=-1)
                        cache_counter = torch.cat((cache_counter.view(-1, cache_counter.shape[-1])[mask].view(num_layers, num_heads, -1), (torch.arange(stride)-stride+1).view(1, 1, -1).repeat(num_layers, num_heads, 1).flip(dims=(2,)).to(self.device)), dim=-1)
                    elif mode == 'tova':
                        eviction_ids = torch.topk(cache_attn_scores[:, :, sink_length:-recent_window], dim=-1, k=stride, largest=False)[1] + sink_length
                        past_key_values = truncate_kv_cache_liso(past_key_values, eviction_ids)
                        _index = torch.ones(num_layers, num_heads, cache_attn_scores.shape[-1], device=self.device).view(num_layers*num_heads, -1)
                        _src = torch.zeros(num_layers, num_heads, stride, device=self.device).view(num_layers*num_heads, -1)
                        mask = _index.scatter(dim=-1, index=eviction_ids.view(num_layers*num_heads, -1), src=_src).bool()
                        cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, stride, device=self.device)), dim=-1)
                    elif mode == 'recency':
                        evict_id = sink_length-stride
                        past_key_values = truncate_kv_cache(past_key_values, start=evict_id, end=evict_id+stride)
                    elif mode == 'random':
                        scores = torch.rand(cache_attn_scores.shape[-1]).to(self.device)
                        scores[-stride:] = -1e9
                        _, evict_id = torch.topk(scores, k=1, dim=-1)
                        evict_id = evict_id[0].cpu().item()
                        past_key_values = truncate_kv_cache(past_key_values, start=evict_id, end=evict_id+stride)
                cur_pos_id += stride
        cur_pos_id = input_ids.shape[-1]
        _tmp = past_key_values[0][0].shape[2]
        print(f"KV cache budget ratio: {_tmp / input_ids.shape[-1]*100:.2f}%({_tmp}/{input_ids.shape[-1]})")
        n = 0
        output_ids = []
        while n < max_new_tokens:
            next_token = torch.multinomial(prob_prev_step, num_samples=1)
            output_ids.append(next_token[0, 0].cpu().item())
            next_token_prob = torch.gather(raw_prob_prev_step, -1, next_token) # (bsz, 1)
            n += 1
            if output_ids[-1] == tokenizer.eos_token_id: break
            outputs = self(input_ids=next_token, 
                            past_key_values=past_key_values,
                            attention_mask=torch.ones(next_token.shape[0], 1+past_key_values[0][0].shape[2], dtype=torch.long, device=self.device),
                            position_ids=torch.LongTensor([cur_pos_id]).to(self.device).view(-1, 1),
                            use_cache=True)
            past_key_values = outputs.past_key_values
            logits_prev_step = outputs.logits[:, -1, :]
            prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)
            cur_pos_id += 1
        decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return decoded_output
    elif kv_mode == 'ppl':
        """
        perplexity computation with fixed kv cache
        """
        length = input_ids.shape[-1]
        if length <= budget:
            outputs_prefilling = self(input_ids=input_ids, use_cache=True)
            past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
            logits_prev_step = logits[:, -1, :]
            prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)
            cur_pos_id = past_key_values[0][0].shape[2]
        else:
            budget = int(length * budget)
            for idx in range(budget, -1, -1):
                if (length-idx)%stride==0: break
            prefix = input_ids[:, :idx]
            recent_window = int(budget*recent_ratio)
            decay_factor = math.exp(math.log(0.001) / budget)
            sink_length = 4
            outputs_prefilling = self(input_ids=prefix, use_cache=True, output_attentions=True)
            past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits[:, :-1].view(-1, logits.shape[-1]), prefix[:, 1:].clone().view(-1))
            prefix_prob = torch.exp(-loss).cpu().numpy().tolist()
            logits_prev_step = logits[:, -1, :]
            _, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)
            prefix_token_lst = input_ids[0].cpu().numpy().tolist()
            cache_tokens = prefix[0].cpu().numpy().tolist()
            cache_probs = [1.0] + prefix_prob
            cache_cur_probs = []
            cache_positions = list(range(prefix.shape[-1]))
            outputs_prefilling.attentions = list(outputs_prefilling.attentions)
            for l in range(num_layers):
                bs = outputs_prefilling.attentions[l].shape[0]
                sl = outputs_prefilling.attentions[l].shape[2]
                tl = outputs_prefilling.attentions[l].shape[3]
                outputs_prefilling.attentions[l] = outputs_prefilling.attentions[l].reshape(bs, num_heads, rep_n, sl, tl).mean(dim=2) # (bs, num_kv_heads, sl, tl)
            if 'decay' in mode and not 'prob' in mode:
                cache_attn_scores = h2o_head_decay_score(outputs_prefilling.attentions, decay_factor, self.device, stride)
            else:
                cache_attn_scores, cache_attn_scores_square = h2o_head_score(outputs_prefilling.attentions, self.device, stride)
            del outputs_prefilling
            torch.cuda.empty_cache()
            cache_counter = torch.tensor([[[1.0]*(idx+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
            cache_counter = torch.cumsum(cache_counter, dim=-1).flip(dims=(2,)) - float(stride)
            cache_counter_token = torch.tensor([1.0]*(idx+stride), device=self.device)
            cache_counter_token = torch.cumsum(cache_counter_token, dim=-1).flip(dims=(0, )) - float(stride)
            n = 0
            output_ids = []
            token_probs = []
            cur_pos_id = past_key_values[0][0].shape[2]
            evicted_positions = []
            log_probs = []
            for token_i in range(idx, length, stride):
                n += stride
                outputs = self(input_ids=input_ids[:, token_i:token_i+stride],
                                past_key_values=past_key_values,
                                attention_mask=torch.ones(1, stride+past_key_values[0][0].shape[2], dtype=torch.long, device=self.device),
                                position_ids=torch.LongTensor(list(range(cur_pos_id, cur_pos_id+stride))).to(self.device).view(1, -1),
                                use_cache=True,
                                output_attentions=True)
                past_key_values = outputs.past_key_values
                logits_prev_step = outputs.logits[:, -1, :]
                log_probs.append(-raw_prob_prev_step.gather(dim=-1, index=input_ids[:, token_i].unsqueeze(-1)).log().cpu().item())
                n_log_probs = torch.nn.CrossEntropyLoss(reduction='none')(outputs.logits[:, :-1].view(-1, logits_prev_step.shape[-1]), input_ids[:, token_i+1:token_i+stride].view(-1)).view(-1)
                log_probs.extend(n_log_probs.cpu().numpy().tolist())
                cache_cur_probs.append(torch.exp(-entropy(raw_prob_prev_step))[0].cpu().item())
                prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)

                # Unified processing for MQA, GQA and MHA
                outputs.attentions = list(outputs.attentions)
                for l in range(num_layers):
                    bs = outputs.attentions[l].shape[0]
                    sl = outputs.attentions[l].shape[2]
                    tl = outputs.attentions[l].shape[3]
                    outputs.attentions[l] = outputs.attentions[l].reshape(bs, num_heads, rep_n, sl, tl).mean(dim=2) # (bs, num_kv_heads, sl, tl)

                # update 
                for pos_ in range(stride): cache_positions.append(cur_pos_id+pos_)
                # update accumulated attention scores
                if 'h2o_head' == mode or 'h2o_head_avg' == mode:
                    for l in range(num_layers):
                        attention_map = outputs.attentions[l][0, :, :, :].sum(dim=1) # (num_heads, stride, stride+l)
                        cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                elif 'h2o_head_std' == mode or 'h2o_head_std_avg' == mode:
                    for l in range(num_layers):
                        attention_map = outputs.attentions[l][0, :, :, :].sum(dim=1) # (num_heads, l)
                        attention_map_sq = ((outputs.attentions[l][0, :, :, :])**2).sum(dim=1)
                        cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                        cache_attn_scores_square[l, :, :attention_map.shape[-1]] += attention_map_sq
                elif 'h2o_head_decay' == mode or 'h2o_head_decay_avg' == mode:
                    for l in range(num_layers):
                        a = decay_factor
                        attention_map = outputs.attentions[l][0, :, :, :].sum(dim=1) # (num_heads, l)
                        cache_attn_scores[l, :, :attention_map.shape[-1]] = a * cache_attn_scores[l, :, :attention_map.shape[-1]] + (1-a) * attention_map
                elif 'tova' == mode:
                    for l in range(num_layers):
                        attention_map = outputs.attentions[l][0, :, -1, :] # (num_heads, l)
                        cache_attn_scores[l, :, :attention_map.shape[-1]] = attention_map
                # evict if current kv cache size exceeds the budget
                cur_kv_size = past_key_values[0][0].shape[2]
                if mode != 'full':
                    cache_counter += float(stride)
                    cache_counter_token += float(stride)
                    positions_tensor = torch.tensor(cache_positions, device=self.device).float()
                    positions_tensor = positions_tensor / float(cur_pos_id)
                    if mode in ['h2o_head', 'h2o_head_decay', 'h2o_head_avg', 'h2o_head_decay_avg', 'h2o_head_prob', 'h2o_head_prob_avg', 'h2o_head_probv2', 'h2o_head_probv2_avg', 'h2o_head_decay_prob', 'h2o_head_decay_probv2', 'h2o_head_decay_prob_avg', 'h2o_head_decay_probv2_avg']:
                        if not 'avg' in mode:
                            eviction_ids = torch.topk(cache_attn_scores[:, :, sink_length:-recent_window], dim=-1, k=stride, largest=False)[1] + sink_length
                        else:
                            eviction_ids = torch.topk(cache_attn_scores[:, :, sink_length:-recent_window] / cache_counter[:, :, sink_length:-recent_window], dim=-1, k=stride, largest=False)[1] + sink_length
                        past_key_values = truncate_kv_cache_liso(past_key_values, eviction_ids)
                        _index = torch.ones(num_layers, num_heads, cache_attn_scores.shape[-1], device=self.device).view(num_layers*num_heads, -1)
                        _src = torch.zeros(num_layers, num_heads, stride, device=self.device).view(num_layers*num_heads, -1)
                        mask = _index.scatter(dim=-1, index=eviction_ids.view(num_layers*num_heads, -1), src=_src).bool()
                        cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, stride, device=self.device)), dim=-1)
                        cache_counter = torch.cat((cache_counter.view(-1, cache_counter.shape[-1])[mask].view(num_layers, num_heads, -1), (torch.arange(stride)-stride+1).view(1, 1, -1).repeat(num_layers, num_heads, 1).flip(dims=(2,)).to(self.device)), dim=-1)
                    elif mode in ['h2o_head_std', 'h2o_head_std_avg', 'h2o_head_probv2_std', 'h2o_head_probv2_std_avg']:
                        cur_std = torch.sqrt(cache_attn_scores_square / cache_counter - (cache_attn_scores / cache_counter)**2)
                        cur_std[:, :, -10:] = 1e9
                        cur_std[:, :, :sink_length] = 1e9
                        _, feasible_ids = torch.topk(cur_std, largest=False, k=max(budget-recent_window-sink_length, stride), dim=-1) # (layers, heads, k)
                        if 'avg' in mode:
                            argmin_id = torch.topk(cache_attn_scores.gather(dim=-1, index=feasible_ids) / cache_counter.gather(dim=-1, index=feasible_ids), dim=-1, largest=False, k=stride)[1] # (layers, heads)
                        else:
                            argmin_id = torch.topk(cache_attn_scores.gather(dim=-1, index=feasible_ids), dim=-1, largest=False, k=stride)[1] # (layers, heads)
                        eviction_ids = feasible_ids.gather(dim=-1, index=argmin_id)
                        past_key_values = truncate_kv_cache_liso(past_key_values, eviction_ids)
                        _index = torch.ones(num_layers, num_heads, cache_attn_scores.shape[-1], device=self.device).view(num_layers*num_heads, -1)
                        _src = torch.zeros(num_layers, num_heads, stride, device=self.device).view(num_layers*num_heads, -1)
                        mask = _index.scatter(dim=-1, index=eviction_ids.view(num_layers*num_heads, -1), src=_src).bool()
                        cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, stride, device=self.device)), dim=-1)
                        cache_attn_scores_square = torch.cat((cache_attn_scores_square.view(-1, cache_attn_scores_square.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, stride, device=self.device)), dim=-1)
                        cache_counter = torch.cat((cache_counter.view(-1, cache_counter.shape[-1])[mask].view(num_layers, num_heads, -1), (torch.arange(stride)-stride+1).view(1, 1, -1).repeat(num_layers, num_heads, 1).flip(dims=(2,)).to(self.device)), dim=-1)
                    elif mode == 'tova':
                        eviction_ids = torch.topk(cache_attn_scores[:, :, sink_length:-recent_window], dim=-1, k=stride, largest=False)[1] + sink_length
                        past_key_values = truncate_kv_cache_liso(past_key_values, eviction_ids)
                        _index = torch.ones(num_layers, num_heads, cache_attn_scores.shape[-1], device=self.device).view(num_layers*num_heads, -1)
                        _src = torch.zeros(num_layers, num_heads, stride, device=self.device).view(num_layers*num_heads, -1)
                        mask = _index.scatter(dim=-1, index=eviction_ids.view(num_layers*num_heads, -1), src=_src).bool()
                        cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, stride, device=self.device)), dim=-1)
                    elif mode == 'recency':
                        evict_id = sink_length-stride
                        past_key_values = truncate_kv_cache(past_key_values, start=evict_id, end=evict_id+stride)
                        evicted_positions.append(cache_positions[evict_id])
                    elif mode == 'random':
                        scores = torch.rand(*positions_tensor.shape).to(self.device)
                        scores[-stride:] = -1e9
                        _, evict_id = torch.topk(scores, k=1, dim=-1)
                        evict_id = evict_id[0].cpu().item()
                        past_key_values = truncate_kv_cache(past_key_values, start=evict_id, end=evict_id+stride)
                        evicted_positions.append(cache_positions[evict_id])
                        for _ in range(stride):
                            cache_positions.pop(evict_id)
                cur_pos_id += stride
        cur_pos_id = input_ids.shape[-1]
        _tmp = past_key_values[0][0].shape[2]
        print(f"KV cache budget ratio: {_tmp / input_ids.shape[-1]*100:.2f}%({_tmp}/{input_ids.shape[-1]})")
        ppl = math.exp(statistics.mean(log_probs))
        return past_key_values, cur_pos_id, ppl

def enable_fixed_kv(model, tokenizer, mode, stride=1):
    model.tokenizer = tokenizer
    import functools
    model.easykv_generate = functools.partial(generate, self=model, kv_mode=mode, stride=stride)
    print(f"Fixed KV Cache for {mode} enabled")