import torch
from typing import Tuple
import math
from tqdm.auto import tqdm, trange
import statistics
import math

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

@torch.inference_mode()
def fixed_budget_v2_ppl(document: str, model, tokenizer, gen_kwargs: dict, budget, ratio, mode:str='typicality'):
    """
    Auto-regressive decoding with fixed kv cache budget
    """
    print(f"Budget KV Cache: {budget}")
    black_list = set()
    top_p = gen_kwargs.get('top_p', 1.0)
    temperature = gen_kwargs.get('temperature', 1.0)

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    input_ids = torch.tensor([tokenizer.bos_token_id], dtype=torch.long, device=model.device).view(1, 1)
    all_inputs = tokenizer([document], return_tensors='pt').to(model.device)
    # budget = int(all_inputs.input_ids.shape[-1]*budget)
    continuation_ids = all_inputs.input_ids[:, input_ids.shape[-1]:][0].cpu().numpy().tolist()
    outputs_prefilling = model(input_ids=input_ids, use_cache=True)
    prefix_token_lst = input_ids[0].cpu().numpy().tolist()
    past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
    logits_prev_step = logits[:, -1, :]
    _, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)

    cache_tokens = []
    cache_probs = []
    cache_typical_probs = []
    cache_cur_probs = []
    cache_positions = []
    cache_attn_scores = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=model.device)
    cache_low_attn_counter = torch.tensor([[[0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=model.device)
    n = 0
    output_ids = []
    token_probs = []
    cur_pos_id = past_key_values[0][0].shape[2]
    evicted_positions = []
    log_probs = []
    for next_token in tqdm(continuation_ids, desc=mode):
        output_ids.append(next_token)
        next_token_prob = torch.gather(raw_prob_prev_step, -1, torch.tensor(next_token, device=model.device).view(1, 1)) # (bsz, 1)
        log_probs.append((-next_token_prob[0, 0].log().cpu().item()))
        token_probs.append((tokenizer.convert_ids_to_tokens([output_ids[-1]])[0], next_token_prob[0, 0].cpu().item()))
        n += 1
        if output_ids[-1] == tokenizer.eos_token_id: break
        outputs = model(input_ids=torch.tensor(next_token, device=model.device).view(1, 1),
                        past_key_values=past_key_values,
                        attention_mask=torch.ones(1, 1+past_key_values[0][0].shape[2], dtype=torch.long, device=model.device),
                        position_ids=torch.LongTensor([cur_pos_id]).to(model.device).view(-1, 1),
                        use_cache=True,
                        output_attentions=True)
        past_key_values = outputs.past_key_values
        logits_prev_step = outputs.logits[:, -1, :]
        cache_cur_probs.append(torch.exp(-entropy(raw_prob_prev_step))[0].cpu().item())
        prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)

        # update 
        cache_probs.append(next_token_prob[0,0].cpu().item())
        cache_tokens.append(output_ids[-1])
        cache_typical_probs.append(torch.exp(-entropy(raw_prob_prev_step))[0].cpu().item())
        cache_positions.append(cur_pos_id)

        # update accumulated attention scores
        if 'h2o' == mode:
            for l in range(num_layers):
                attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
        elif 'h2o_decay' == mode:
            a = 0.96
            for l in range(num_layers):
                attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                cache_attn_scores[l, :, :attention_map.shape[-1]] = a * cache_attn_scores[l, :, :attention_map.shape[-1]] + (1-a) * attention_map
        elif 'scissor' == mode:
            for l in range(num_layers):
                attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                attention_map = attention_map / attention_map.sum(dim=-1, keepdims=True)
                threshold = 1 / attention_map.shape[-1]
                cache_low_attn_counter[l, :, :attention_map.shape[-1]] +=  (attention_map >= torch.tensor(threshold, device=model.device)).to(torch.long)
        elif 'scissor_decay' == mode:
            a = 0.95
            for l in range(num_layers):
                attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                attention_map = attention_map / attention_map.sum(dim=-1, keepdims=True)
                threshold = 1 / attention_map.shape[-1]
                cache_low_attn_counter[l, :, :attention_map.shape[-1]] = a*cache_low_attn_counter[l, :, :attention_map.shape[-1]] + (1-a)*(attention_map >= torch.tensor(threshold, device=model.device)).to(torch.long)
        # evict if current kv cache size exceeds the budget
        cur_kv_size = past_key_values[0][0].shape[2]
        if (cur_kv_size-len(prefix_token_lst)) > budget and mode != 'full':
            in_blk_lst = torch.tensor([token_id in black_list for token_id in cache_tokens], dtype=torch.float, device=model.device) * -1e9
            probs_tensor = torch.tensor(cache_probs, device=model.device)
            positions_tensor = torch.tensor(cache_positions, device=model.device).float()
            positions_tensor = positions_tensor / float(cur_pos_id)
            recent_ratio = ratio
            recent_window = int(budget*recent_ratio)
            if mode == 'probability':
                scores = probs_tensor
                retain_cnt = int(budget*recent_ratio)
                scores[-retain_cnt:] = -1e9
                scores += in_blk_lst
                _, evict_id = torch.topk(scores, k=1, dim=-1)
                evict_id = evict_id[0].cpu().item()
                past_key_values = truncate_kv_cache(past_key_values, start=len(prefix_token_lst)+evict_id, end=len(prefix_token_lst)+evict_id+1)
                evicted_positions.append(cache_positions[evict_id]-len(prefix_token_lst))
                cache_probs.pop(evict_id)
                cache_tokens.pop(evict_id)
                cache_typical_probs.pop(evict_id)
                cache_cur_probs.pop(evict_id)
                cache_positions.pop(evict_id)
            elif mode == 'scissor' or mode == 'scissor_decay':
                eviction_ids = [[None for _ in range(num_heads)] for _ in range(num_layers)]
                for l in range(num_layers):
                    for h in range(num_heads):
                        if recent_window>0:
                            eviction_ids[l][h] = torch.argmin(cache_attn_scores[l][h][:-recent_window]).cpu().item() + len(prefix_token_lst)
                        else: eviction_ids[l][h] = torch.argmin(cache_attn_scores[l][h]).cpu().item() + len(prefix_token_lst)
                past_key_values = truncate_kv_cache_liso(past_key_values, eviction_ids)
                for l in range(num_layers):
                    for h in range(num_heads):
                        _tmp = cache_low_attn_counter[l, h, eviction_ids[l][h]-len(prefix_token_lst)+1:].clone()
                        cache_low_attn_counter[l, h, eviction_ids[l][h]-len(prefix_token_lst):-1] = _tmp
                        cache_low_attn_counter[l, h, -1] = 0
            elif mode == 'h2o' or mode == 'h2o_decay':
                eviction_ids = [[None for _ in range(num_heads)] for _ in range(num_layers)]
                for l in range(num_layers):
                    for h in range(num_heads):
                        if recent_window>0:
                            eviction_ids[l][h] = torch.argmin(cache_attn_scores[l][h][:-recent_window]).cpu().item() + len(prefix_token_lst)
                        else:
                            eviction_ids[l][h] = torch.argmin(cache_attn_scores[l][h]).cpu().item() + len(prefix_token_lst)
                past_key_values = truncate_kv_cache_liso(past_key_values, eviction_ids)
                for l in range(num_layers):
                    for h in range(num_heads):
                        _tmp = cache_attn_scores[l, h, eviction_ids[l][h]-len(prefix_token_lst)+1:].clone()
                        cache_attn_scores[l, h, eviction_ids[l][h]-len(prefix_token_lst):-1] = _tmp
                        cache_attn_scores[l, h, -1] = 0.0
            elif mode == 'recency':
                scores = 1.0 - positions_tensor
                _, evict_id = torch.topk(scores, k=1, dim=-1)
                evict_id = evict_id[0].cpu().item()
                past_key_values = truncate_kv_cache(past_key_values, start=len(prefix_token_lst)+evict_id, end=len(prefix_token_lst)+evict_id+1)
                evicted_positions.append(cache_positions[evict_id]-len(prefix_token_lst))
                cache_probs.pop(evict_id)
                cache_tokens.pop(evict_id)
                cache_typical_probs.pop(evict_id)
                cache_cur_probs.pop(evict_id)
                cache_positions.pop(evict_id)
            elif mode == 'random':
                scores = torch.rand(*positions_tensor.shape).to(model.device)
                _, evict_id = torch.topk(scores, k=1, dim=-1)
                evict_id = evict_id[0].cpu().item()
                past_key_values = truncate_kv_cache(past_key_values, start=len(prefix_token_lst)+evict_id, end=len(prefix_token_lst)+evict_id+1)
                evicted_positions.append(cache_positions[evict_id]-len(prefix_token_lst))
                cache_probs.pop(evict_id)
                cache_tokens.pop(evict_id)
                cache_typical_probs.pop(evict_id)
                cache_cur_probs.pop(evict_id)
                cache_positions.pop(evict_id)
        cur_pos_id += 1
    print('\n')
    if len(log_probs)>budget: log_probs = log_probs[budget:]
    ppl = math.exp(statistics.mean(log_probs))
    return past_key_values, cur_pos_id, ppl

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
        cache_typical_probs = []
        cache_cur_probs = []
        cache_positions = []
        cache_attn_scores = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
        cache_attn_scores_decay_avg_std = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
        cache_attn_scores_square = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
        cache_attn_scores_binary = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
        cache_attn_scores_square_binary = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
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
            cache_typical_probs.append(torch.exp(-entropy(raw_prob_prev_step))[0].cpu().item())
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
            elif 'h2o_head_std_avg_binary' == mode or 'h2o_head_std_avg_binary_dynamic' == mode:
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                    # renorm_attention_map = attention_map / attention_map.sum(dim=-1, keepdims=True)
                    renorm_attention_map = attention_map
                    if not 'dynamic' in mode:
                        threshold = 1.0 / (renorm_attention_map.shape[-1]+len(prefix_token_lst))
                    else:
                        threshold = torch.exp(-entropy(outputs.attentions[l][0, :, 0, :])).unsqueeze(-1) # (num_heads, 1)
                    cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                    cache_attn_scores_binary[l, :, :attention_map.shape[-1]] += (renorm_attention_map>=threshold).float()
                    cache_attn_scores_square_binary[l, :, :attention_map.shape[-1]] += ((renorm_attention_map>=threshold).float()) ** 2
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
            elif 'h2o_head_decay_avg_std_binary' == mode or 'h2o_head_decay_avg_std_binary_dynamic' == mode:
                a = 0.96
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                    cache_attn_scores[l, :, :attention_map.shape[-1]] = a * cache_attn_scores[l, :, :attention_map.shape[-1]] + (1-a) * attention_map
                    renorm_attention_map = attention_map
                    if not 'dynamic' in mode:
                        threshold = 1.0 / (renorm_attention_map.shape[-1]+len(prefix_token_lst))
                    else:
                        threshold = torch.exp(-entropy(outputs.attentions[l][0, :, 0, :])).unsqueeze(-1) # (num_heads, 1)
                    cache_attn_scores_binary[l, :, :attention_map.shape[-1]] += (renorm_attention_map>=threshold).float()
                    cache_attn_scores_square_binary[l, :, :attention_map.shape[-1]] += ((renorm_attention_map>=threshold).float()) ** 2
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
                    cache_typical_probs.pop(evict_id)
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
                    cache_typical_probs.pop(evict_id)
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
                    cache_typical_probs.pop(evict_id)
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
        if length <= budget:
            outputs_prefilling = self(input_ids=input_ids, use_cache=True)
            past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
            logits_prev_step = logits[:, -1, :]
            prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)
            cur_pos_id = past_key_values[0][0].shape[2]
        else:
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
            cache_probs = [1.0] + prefix_prob
            cache_typical_probs = []
            cache_cur_probs = []
            cache_positions = list(range(prefix.shape[-1]))
            cache_attn_scores_token = outputs_prefilling.attentions[-1][0].mean(dim=0).sum(dim=0).cpu().numpy().tolist()+[0.0] # (budget,)
            cache_attn_scores_binary = torch.tensor([[[0.0]*(idx+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
            cache_attn_scores_square_binary = torch.tensor([[[0.0]*(idx+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
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
                cache_typical_probs.append(torch.exp(-entropy(raw_prob_prev_step))[0].cpu().item())
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
                    positions_tensor = torch.tensor(cache_positions, device=self.device).float()
                    positions_tensor = positions_tensor / float(cur_pos_id)
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
                        _, feasible_ids = torch.topk(cur_std, largest=False, k=budget-recent_window-sink_length, dim=-1) # (layers, heads, k)
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
        n = 0
        output_ids = []
        token_probs = []
        while n < max_new_tokens:
            next_token = torch.multinomial(prob_prev_step, num_samples=1)
            output_ids.append(next_token[0, 0].cpu().item())
            next_token_prob = torch.gather(raw_prob_prev_step, -1, next_token) # (bsz, 1)
            token_probs.append((tokenizer.convert_ids_to_tokens([output_ids[-1]])[0], next_token_prob[0, 0].cpu().item()))
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
            cache_typical_probs = []
            cache_cur_probs = []
            cache_positions = list(range(prefix.shape[-1]))
            cache_attn_scores_token = outputs_prefilling.attentions[-1][0].mean(dim=0).sum(dim=0).cpu().numpy().tolist()+[0.0] # (budget,)
            cache_attn_scores_binary = torch.tensor([[[0.0]*(idx+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
            cache_attn_scores_square_binary = torch.tensor([[[0.0]*(idx+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
            if 'decay' in mode and not 'prob' in mode:
                cache_attn_scores = h2o_head_decay_score(outputs_prefilling.attentions, decay_factor, self.device, stride)
            else:
                cache_attn_scores, cache_attn_scores_square = h2o_head_score(outputs_prefilling.attentions, self.device, stride)
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
                cache_typical_probs.append(torch.exp(-entropy(raw_prob_prev_step))[0].cpu().item())
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
                        _, feasible_ids = torch.topk(cur_std, largest=False, k=budget-recent_window-sink_length, dim=-1) # (layers, heads, k)
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
    model.generate = functools.partial(generate, self=model, kv_mode=mode, stride=stride)
    print(f"Fixed KV Cache for {mode} enabled")

if __name__ == '__main__':
    from transformers import (AutoModelForCausalLM, AutoTokenizer)
    # define the model path and the corresponding prompt template
    MODEL_CONFIGS = {
        'wizardlm_13b': dict(path='/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--WizardLM--WizardLM-13B-V1.2/snapshots/cf5f40382559f19e13874e45b39575171ca46ef8', template="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Hello!\nASSISTANT: Hello!</s>\nUSER: {inst}\nASSISTANT:"),
        'llama2_13b_chat': dict(path='/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496/', template="[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n{inst}[/INST]"),
        'vicuna_13b': dict(path='/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--lmsys--vicuna-13b-v1.5/snapshots/3deb0106f72a3a433f0c6ea0cb978bdf14bcd3a6/', template="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Hello!\nASSISTANT: Hello!</s>\nUSER: {inst}\nASSISTANT:"),
        'openchat': dict(path='/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--openchat--openchat_v3.2_super/snapshots/aab7ce4d48b31a295a0116b61569d8e87a09bb7a/', template="GPT4 User: {inst}<|end_of_turn|>GPT4 Assistant:"),
        'vicuna_7b': dict(path='/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5/', template="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: Hello!\nASSISTANT: Hello!</s>\nUSER: {inst}\nASSISTANT:"),
        'wizardlm_7b': dict(path='/cpfs01/user/rensiyu/language_modeling/stanford_alpaca/output_mle_fp16_recycledWiz70k_llama2_7b_512', template="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Response:"),
        'alpaca_7b': dict(path='/cpfs01/user/rensiyu/language_modeling/stanford_alpaca/output_mle_recycledAlpaca52k_llama2_7b_512_ds', template="Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Response:"),
        'zephyr_7b': dict(path='/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/dc24cabd13eacd3ae3a5fe574bd645483a335a4a/', template="<|system|>\nYou are a friendly chatbot who always responds in a helpful and detailed manner to the user's questions.</s>\n<|user|>\n{inst}</s>\n<|assistant|>\n"),
        'llama2_7b_chat': dict(path='/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-chat-hf/', template="[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n{inst}[/INST]"),
        'llama2_7b': dict(path='/cpfs01/shared/public/public_hdd/llmeval/model_weights/llama2/model_weights_hf/llama-2-7b-hf'),
        'llama2_13b': dict(path='/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--meta-llama--Llama-2-13b-hf/snapshots/dc1d3b3bfdb69df26f8fc966c16353274b138c55/'),
    }
    model_name = 'zephyr_7b'
    path = MODEL_CONFIGS[model_name]['path']
    template = MODEL_CONFIGS[model_name]['template']
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map='auto').eval()
    tokenizer = AutoTokenizer.from_pretrained(path)

    # =============== Turn on Fixed KV Cache ==================
    # =============== Here is an example of long prompt encoding mode ===================
    for stride in [1,2,3,4,5]:
        # Setup KV caching mode
        enable_fixed_kv(model, tokenizer, mode='encoding', stride=stride)

        # Test input
        article = "###\nArticle: It was the first time the Single Transferable Vote (STV) system had been used to select two members in the same ward in a by-election. The SNP topped the vote in the Leith Walk by-election, while Scottish Labour won the second seat from the Greens. The by-election was called after Deidre Brock of the SNP and Maggie Chapman of the Scottish Greens stood down. The SNP's John Lewis Ritchie topped the Leith Walk poll with 2,290 votes. He was elected at stage one in the STV process with a swing in first-preference votes of 7.6% from Labour. Labour's Marion Donaldson received 1,623 votes, ahead of Susan Jane Rae of the Scottish Greens on 1,381. Ms Donaldson was elected at stage 10 of the voting process after other preferences had been considered. The by-election was called after Ms Brock stood down when she was elected as the SNP MP for Edinburgh North and Leith in May. Ms Chapman, of the Scottish Greens, resigned from her post to concentrate on standing for the Scottish Parliament in next May's election. The turnout for the by-election was 25.1%. The SNP also held the Midlothian West seat on Midlothian Council with a swing of 6.3% from Labour. The party's Kelly Parry secured 1,540 votes, ahead of Labour's Ian Miller on 945 votes. The by-election was called after Owen Thompson was elected as SNP MP for the Midlothian constituency.\n\nSummarize the above article in 1 sentence.\n"
        prompt = f"Write a SHORT summary of the following text delimited by triple backticks. Return your response which covers the key points of the text.\n```{article}```"
        input_prompt = template.format(inst=prompt)

        # Inference with fixed kv cache applied to prompt encoding phase
        # define eviction policy
        kv_policy = 'h2o_head_std_avg'
        # define sampling parameters
        gen_kwargs = dict(
            temperature=1e-9,
            top_p=1.0,
            max_new_tokens=256,
            budget=0.5,
            kv_policy=kv_policy
        )
        input_ids = tokenizer([input_prompt], return_tensors='pt').input_ids.to(model.device)
        output = model.generate(input_ids=input_ids, generation_config=gen_kwargs)
        print(f"{'='*20} {kv_policy} {'='*20}\n{output}")

    # ============= Turn on Fixed KV Cache ====================
    # ============= Here is an example of decoding mode ==================
    # enable_fixed_kv(model, tokenizer, mode='decoding', stride=1)
    # prompt = f"What are the names of some famous actors that started their careers on Broadway?"
    # input_prompt = template.format(inst=prompt)
    # kv_policy = 'h2o_head_decay_avg_std'
    # gen_kwargs = dict(
    #     temperature=1e-9,
    #     top_p=1.0,
    #     max_new_tokens=2048,
    #     budget=200,
    #     kv_policy=kv_policy
    # )
    # input_ids = tokenizer([input_prompt], return_tensors='pt').input_ids.to(model.device)
    # output = model.generate(input_ids=input_ids, generation_config=gen_kwargs)
    # print(f"{'='*20} {kv_policy} {'='*20}\n{output}")