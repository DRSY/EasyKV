import torch
from typing import Tuple
import math
import statistics
from functools import partial
from .utils import modify_method_of_instance
from .llama_patch import llama_forward, llama_forward_stream
from .mistral_patch import mistral_forward, mistral_forward_stream

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
    # print("Reserved GPU memory:", round(memory_stats["reserved_bytes.all.allocated"]/(1024**3), 3), "GB")


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

def truncate_kv_cache_liso_mean(kv_cache, eviction_ids):
    """
    eviction_ids: (num_layers, num_heads, k+1)
    """
    kv_cache = list(kv_cache)
    for i in range(len(kv_cache)):
        kv_cache[i] = list(kv_cache[i])
    l = kv_cache[0][0].shape[2]
    head_dim = kv_cache[0][0].shape[-1]
    num_heads = kv_cache[0][0].shape[1]
    for i in range(eviction_ids.shape[0]):
        src_ = torch.zeros(num_heads, eviction_ids.shape[-1]).to(kv_cache[0][0].device)
        mask = torch.ones(num_heads, l, device=kv_cache[0][0].device).scatter(dim=-1, index=eviction_ids[i], src=src_).bool()
        evicted_mask = ~mask
        key_evicted_mean = torch.mean(kv_cache[i][0][0][evicted_mask, ...].view(1, num_heads, -1, head_dim), dim=2, keepdim=True)
        value_evicted_mean = torch.mean(kv_cache[i][1][0][evicted_mask, ...].view(1, num_heads, -1, head_dim), dim=2, keepdim=True)
        kv_cache[i][0] = torch.cat((kv_cache[i][0][0][mask, ...].view(1, num_heads, -1, head_dim), key_evicted_mean), dim=2)
        kv_cache[i][1] = torch.cat((kv_cache[i][1][0][mask, ...].view(1, num_heads, -1, head_dim), value_evicted_mean), dim=2)
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

def h2o_head_score(attention_map, device, stride, budget, num_layers, num_heads, empty=False):
    # if attention_map is not None:
    #     attention_map = list(attention_map)
    #     num_layers = len(attention_map)
    #     budget = attention_map[0].shape[-1]
    cache_attn_scores = torch.tensor([[[0.0]*(budget+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=device)
    cache_attn_scores_square = torch.tensor([[[0.0]*(budget+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=device)
    if not empty:
        for l in range(num_layers):
            attention_map[l] = attention_map[l].to('cuda')
            cache_attn_scores[l, :, :attention_map[l].shape[-1]] = torch.sum(attention_map[l][0], dim=1)
            cache_attn_scores_square[l, :, :attention_map[l].shape[-1]] = torch.sum(attention_map[l][0]**2, dim=1)
            attention_map[l] = None
    return cache_attn_scores, cache_attn_scores_square

def process_for_mqa_gqa(attentions, num_layers, num_heads, rep_n):
    # Unified processing for MQA, GQA and MHA
    attentions = list(attentions)
    for l in range(num_layers):
        bs = attentions[l].shape[0]
        sl = attentions[l].shape[2]
        tl = attentions[l].shape[3]
        attentions[l] = attentions[l].reshape(bs, num_heads, rep_n, sl, tl).mean(dim=2) # (bs, num_kv_heads, sl, tl)
    return attentions


@torch.inference_mode()
def generate(self, input_ids, generation_config, kv_mode='encoding', stride=1, report_decoding_latency: bool=False):
    temperature = generation_config.get('temperature', 1.0)
    top_p = generation_config.get('top_p', 1.0)
    max_new_tokens = generation_config.get('max_new_tokens', 1024)
    budget = generation_config.get('budget', 0.5)
    mode = generation_config.get('kv_policy', 'recency')
    temp_length = generation_config.get('temp_length', 4)
    recent_ratio = generation_config.get('recent_ratio', 0.1)
    keep_attention = generation_config.get('keep_attention', False)
    eos_token_ids = generation_config.get('eos_token_ids', [self.tokenizer.eos_token_id])
    streaming = generation_config.get('streaming', False)
    num_layers = self.config.num_hidden_layers
    if not hasattr(self.config, "num_key_value_heads"): num_heads = self.config.num_attention_heads
    else: num_heads = self.config.num_key_value_heads
    tokenizer = self.tokenizer
    # Handle MQA and GQA
    is_gqa = hasattr(self.config, "num_key_value_heads") and getattr(self.config, "num_key_value_heads") != getattr(self.config, "num_attention_heads")
    if is_gqa: rep_n = self.config.num_attention_heads // self.config.num_key_value_heads
    else: rep_n = 1
    length = input_ids.shape[-1]
    if kv_mode == 'auto':
        length = input_ids.shape[-1]
        assert type(budget) == int
        if budget > length:
            kv_mode = 'decoding'
            budget -= length
        else:
            kv_mode = 'encoding_decoding'
    if kv_mode == 'decoding':
        """
        auto-regressive decoding
        """
        outputs_prefilling = self(input_ids=input_ids, use_cache=True)
        prefix_token_lst = input_ids[0].cpu().numpy().tolist()
        past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
        logits_prev_step = logits[:, -1, :]
        prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)

        cache_tokens = []
        cache_probs = []
        cache_cur_probs = []
        cache_positions = []
        cache_attn_scores = torch.tensor([[[0.0]*(budget+1) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
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
        if 'llama' in self.config.architectures[0].lower():
            modify_method_of_instance(self, "LlamaAttention", "forward", partial(llama_forward if not streaming else llama_forward_stream, attn_device='cuda'))
        else:
            modify_method_of_instance(self, "MistralAttention", "forward", partial(mistral_forward if not streaming else mistral_forward_stream, attn_device='cuda'))
        while n < max_new_tokens:
            next_token = torch.multinomial(prob_prev_step, num_samples=1)
            output_ids.append(next_token[0, 0].cpu().item())
            next_token_prob = torch.gather(raw_prob_prev_step, -1, next_token) # (bsz, 1)
            token_probs.append((tokenizer.convert_ids_to_tokens([output_ids[-1]])[0], next_token_prob[0, 0].cpu().item()))
            n += 1
            if output_ids[-1] in eos_token_ids: break
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
            if 'h2o_head' == mode:
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                    cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
            elif 'roco' == mode:
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                    cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                    cache_attn_scores_square[l, :, :attention_map.shape[-1]] += attention_map ** 2
            elif 'tova' == mode:
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, len(prefix_token_lst):] # (num_heads, l)
                    cache_attn_scores[l, :, :attention_map.shape[-1]] = attention_map
            # evict if current kv cache size exceeds the budget
            cur_kv_size = past_key_values[0][0].shape[2]
            if (cur_kv_size-len(prefix_token_lst)) > budget and mode != 'full':
                cache_counter += 1.0
                cache_counter_token += 1.0
                positions_tensor = torch.tensor(cache_positions, device=self.device).float()
                positions_tensor = positions_tensor / float(cur_pos_id)
                recent_ratio = 0.3
                recent_window = int(budget*recent_ratio)
                if mode in ['h2o_head']:
                    eviction_ids = torch.argmin(cache_attn_scores[:, :, :-recent_window], dim=-1) + len(prefix_token_lst)
                    _eviction_ids = eviction_ids
                    eviction_ids = eviction_ids.cpu().numpy().tolist()
                    past_key_values = truncate_kv_cache_silo(past_key_values, eviction_ids)
                    _index = torch.arange(cache_attn_scores.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).repeat(num_layers, num_heads, 1)
                    _eviction_ids -= len(prefix_token_lst)
                    mask = (_eviction_ids.unsqueeze(-1)!=_index).view(-1, _index.shape[-1])
                    cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, _index.shape[-1]-1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                elif mode in ['roco']:
                    cur_std = torch.sqrt(cache_attn_scores_square / cache_counter - (cache_attn_scores / cache_counter)**2)
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
        if type(budget) == float and budget >= 1.0 or type(budget) == int and budget >= length:
            outputs_prefilling = self(input_ids=input_ids, use_cache=True)
            past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
            logits_prev_step = logits[:, -1, :]
            prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)
            cur_pos_id = past_key_values[0][0].shape[2]
        else:
            # In case budget is also large, the attention_map will occupy a lot of memory
            # We offload attention_map to CPU first and move it layer by laer to GPU to compute eviction score
            if 'llama' in self.config.architectures[0].lower():
                modify_method_of_instance(self, "LlamaAttention", "forward", partial(llama_forward if not streaming else llama_forward_stream, attn_device='cuda'))
            else:
                modify_method_of_instance(self, "MistralAttention", "forward", partial(mistral_forward if not streaming else mistral_forward_stream, attn_device='cuda'))
            if type(budget) == float:
                budget = int(length * budget) + stride
            elif type(budget) == int: 
                budget += stride
            for idx in range(budget, -1, -1):
                if (length-idx)%stride==0: break
            for r_idx in range(idx-1, -1, -1):
                if (idx-r_idx)%stride==0: break
            prefix = input_ids[:, :r_idx]
            recent_window = int(budget*recent_ratio)
            sink_length = temp_length
            outputs_prefilling = self(input_ids=prefix, use_cache=True, output_attentions=keep_attention)
            past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits_prev_step = logits[:, -1, :]
            _, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)
            prefix_token_lst = input_ids[0].cpu().numpy().tolist()
            cache_tokens = prefix[0].cpu().numpy().tolist()
            if keep_attention:
                outputs_prefilling.attentions = process_for_mqa_gqa(outputs_prefilling.attentions, num_layers, num_heads, rep_n)
            cache_attn_scores, cache_attn_scores_square = h2o_head_score(outputs_prefilling.attentions, self.device, stride, idx, num_layers, num_heads, empty=not keep_attention)
            # Back to GPU
            if 'llama' in self.config.architectures[0].lower():
                modify_method_of_instance(self, "LlamaAttention", "forward", partial(llama_forward if not streaming else llama_forward_stream, attn_device='cuda'))
            else:
                modify_method_of_instance(self, "MistralAttention", "forward", partial(mistral_forward if not streaming else mistral_forward_stream, attn_device='cuda'))

            if keep_attention:
                cache_counter = torch.tensor([[[1.0]*(idx+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
                cache_counter = torch.cumsum(cache_counter, dim=-1).flip(dims=(2,)) - float(stride)
            else:
                cache_counter = torch.tensor([[[float(stride)]*idx+torch.arange(stride, 0, -1).numpy().tolist() for _ in range(num_heads)] for _ in range(num_layers)], device=self.device) - float(stride)
            cache_counter_token = torch.tensor([1.0]*(idx+stride), device=self.device)
            cache_counter_token = torch.cumsum(cache_counter_token, dim=-1).flip(dims=(0, )) - float(stride)
            n = 0
            output_ids = []
            token_probs = []
            cur_pos_id = past_key_values[0][0].shape[2]
            evicted_positions = []
            log_probs = []
            # for token_i in range(idx, length, stride):
            for token_i in range(r_idx, length, stride):
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
                outputs.attentions = process_for_mqa_gqa(outputs.attentions, num_layers, num_heads, rep_n)

                cur_kv_size = past_key_values[0][0].shape[2]
                # update accumulated attention scores
                if cur_kv_size>idx or keep_attention:
                    if 'h2o_head' == mode:
                        for l in range(num_layers):
                            attention_map = outputs.attentions[l][0, :, :, :].sum(dim=1) # (num_heads, stride, stride+l)
                            cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                    elif 'roco' == mode:
                        for l in range(num_layers):
                            attention_map = outputs.attentions[l][0, :, :, :].sum(dim=1) # (num_heads, l)
                            attention_map_sq = ((outputs.attentions[l][0, :, :, :])**2).sum(dim=1)
                            cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                            cache_attn_scores_square[l, :, :attention_map.shape[-1]] += attention_map_sq
                    elif 'tova' == mode:
                        for l in range(num_layers):
                            attention_map = outputs.attentions[l][0, :, -1, :].mean(dim=0).unsqueeze(0).repeat(num_heads, 1) # (num_heads, l)
                            cache_attn_scores[l, :, :attention_map.shape[-1]] = attention_map
                # evict if current kv cache size exceeds the budget
                if mode != 'full' and cur_kv_size>idx:
                    cache_counter += float(stride)
                    cache_counter_token += float(stride)
                    if mode in ['h2o_head']:
                        eviction_ids = torch.topk(cache_attn_scores[:, :, sink_length:-recent_window], dim=-1, k=stride, largest=False)[1] + sink_length
                        past_key_values = truncate_kv_cache_liso(past_key_values, eviction_ids)
                        _index = torch.ones(num_layers, num_heads, cache_attn_scores.shape[-1], device=self.device).view(num_layers*num_heads, -1)
                        _src = torch.zeros(num_layers, num_heads, stride, device=self.device).view(num_layers*num_heads, -1)
                        mask = _index.scatter(dim=-1, index=eviction_ids.view(num_layers*num_heads, -1), src=_src).bool()
                        cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, stride, device=self.device)), dim=-1)
                        cache_counter = torch.cat((cache_counter.view(-1, cache_counter.shape[-1])[mask].view(num_layers, num_heads, -1), (torch.arange(stride)-stride+1).view(1, 1, -1).repeat(num_layers, num_heads, 1).flip(dims=(2,)).to(self.device)), dim=-1)
                    elif mode in ['roco']:
                        cur_std = torch.sqrt(cache_attn_scores_square / cache_counter - (cache_attn_scores / cache_counter)**2)
                        cur_std[:, :, -10:] = 1e9
                        cur_std[:, :, :sink_length] = 1e9
                        _, feasible_ids = torch.topk(cur_std, largest=False, k=max(budget-recent_window-sink_length, stride), dim=-1) # (layers, heads, k)
                        argmin_id = torch.topk(cache_attn_scores.gather(dim=-1, index=feasible_ids) / cache_counter.gather(dim=-1, index=feasible_ids), dim=-1, largest=False, k=stride)[1] # (layers, heads)
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
                        evict_id = sink_length
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
        decoding_times = []
        import time
        while n < max_new_tokens:
            next_token = torch.multinomial(prob_prev_step, num_samples=1)
            output_ids.append(next_token[0, 0].cpu().item())
            next_token_prob = torch.gather(raw_prob_prev_step, -1, next_token) # (bsz, 1)
            n += 1
            if output_ids[-1] in eos_token_ids: break
            s = time.time()
            outputs = self(input_ids=next_token, 
                            past_key_values=past_key_values,
                            attention_mask=torch.ones(next_token.shape[0], 1+past_key_values[0][0].shape[2], dtype=torch.long, device=self.device),
                            position_ids=torch.LongTensor([cur_pos_id]).to(self.device).view(-1, 1),
                            use_cache=True)
            past_key_values = outputs.past_key_values
            logits_prev_step = outputs.logits[:, -1, :]
            prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)
            e = time.time()
            cur_step_time = e-s
            decoding_times.append(cur_step_time)
            cur_pos_id += 1
        decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        if report_decoding_latency: print(f"Per-step decoding latency: {statistics.mean(decoding_times[1:]):.3f}")
        return decoded_output
    elif kv_mode == 'encoding_decoding':
        """
        after encoding, budget-1 decoding
        """
        length = input_ids.shape[-1]
        assert type(budget) == int and budget <= length
        white_lst = ['random', 'recency', 'tova', 'roco']
        assert mode in white_lst, f"mode must be within {white_lst}, get {mode} instead"
        # In case budget is also large, the attention_map will occupy a lot of memory
        # We offload attention_map to CPU first and move it layer by layer to GPU to compute eviction score
        if 'llama' in self.config.architectures[0].lower():
            modify_method_of_instance(self, "LlamaAttention", "forward", partial(llama_forward if not streaming else llama_forward_stream, attn_device='cuda'))
        else:
            modify_method_of_instance(self, "MistralAttention", "forward", partial(mistral_forward if not streaming else mistral_forward_stream, attn_device='cuda'))
        if type(budget) == float:
            budget = int(length * budget) + stride
        elif type(budget) == int: 
            budget += stride
            if budget >= length: budget -= stride
        for idx in range(budget, -1, -1):
            if (length-idx)%stride==0: break
        for r_idx in range(1, idx):
            if (idx-r_idx)%stride==0: break
        # prefix = input_ids[:, :idx]
        prefix = input_ids[:, :r_idx]
        recent_window = int(budget*recent_ratio)
        sink_length = temp_length
        outputs_prefilling = self(input_ids=prefix, use_cache=True, output_attentions=keep_attention)
        past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        logits_prev_step = logits[:, -1, :]
        _, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)
        prefix_token_lst = input_ids[0].cpu().numpy().tolist()
        cache_tokens = prefix[0].cpu().numpy().tolist()
        if keep_attention:
            outputs_prefilling.attentions = process_for_mqa_gqa(outputs_prefilling.attentions, num_layers, num_heads, rep_n)
        cache_attn_scores, cache_attn_scores_square = h2o_head_score(outputs_prefilling.attentions, self.device, stride, idx, num_layers, num_heads, empty=not keep_attention)
        # Back to GPU
        if 'llama' in self.config.architectures[0].lower():
            modify_method_of_instance(self, "LlamaAttention", "forward", partial(llama_forward if not streaming else llama_forward_stream, attn_device='cuda'))
        else:
            modify_method_of_instance(self, "MistralAttention", "forward", partial(mistral_forward if not streaming else mistral_forward_stream, attn_device='cuda'))

        if keep_attention:
            cache_counter = torch.tensor([[[1.0]*(idx+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
            cache_counter = torch.cumsum(cache_counter, dim=-1).flip(dims=(2,)) - float(stride)
        else:
            cache_counter = torch.tensor([[[float(stride)]*idx+torch.arange(stride, 0, -1).numpy().tolist() for _ in range(num_heads)] for _ in range(num_layers)], device=self.device) - float(stride)
        cache_counter_token = torch.tensor([1.0]*(idx+stride), device=self.device)
        cache_counter_token = torch.cumsum(cache_counter_token, dim=-1).flip(dims=(0, )) - float(stride)
        n = 0
        output_ids = []
        token_probs = []
        cur_pos_id = past_key_values[0][0].shape[2]
        evicted_positions = []
        log_probs = []
        # for token_i in range(idx, length, stride):
        for token_i in range(r_idx, length, stride):
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
            outputs.attentions = process_for_mqa_gqa(outputs.attentions, num_layers, num_heads, rep_n)

            cur_kv_size = past_key_values[0][0].shape[2]
            if cur_kv_size>idx or keep_attention:
                # update accumulated attention scores
                if 'h2o_head' == mode:
                    for l in range(num_layers):
                        attention_map = outputs.attentions[l][0, :, :, :].sum(dim=1) # (num_heads, stride, stride+l)
                        cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                elif 'roco' == mode:
                    for l in range(num_layers):
                        attention_map = outputs.attentions[l][0, :, :, :].sum(dim=1) # (num_heads, l)
                        attention_map_sq = ((outputs.attentions[l][0, :, :, :])**2).sum(dim=1)
                        cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                        cache_attn_scores_square[l, :, :attention_map.shape[-1]] += attention_map_sq
                elif 'tova' == mode:
                    for l in range(num_layers):
                        attention_map = outputs.attentions[l][0, :, -1, :] # (num_heads, l)
                        cache_attn_scores[l, :, :attention_map.shape[-1]] = attention_map
            # evict if current kv cache size exceeds the budget
            if mode != 'full' and cur_kv_size>idx:
                cache_counter += float(stride)
                cache_counter_token += float(stride)
                if mode in ['h2o_head']:
                    eviction_ids = torch.topk(cache_attn_scores[:, :, sink_length:-recent_window], dim=-1, k=stride, largest=False)[1] + sink_length
                    past_key_values = truncate_kv_cache_liso(past_key_values, eviction_ids)
                    _index = torch.ones(num_layers, num_heads, cache_attn_scores.shape[-1], device=self.device).view(num_layers*num_heads, -1)
                    _src = torch.zeros(num_layers, num_heads, stride, device=self.device).view(num_layers*num_heads, -1)
                    mask = _index.scatter(dim=-1, index=eviction_ids.view(num_layers*num_heads, -1), src=_src).bool()
                    cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, stride, device=self.device)), dim=-1)
                    cache_counter = torch.cat((cache_counter.view(-1, cache_counter.shape[-1])[mask].view(num_layers, num_heads, -1), (torch.arange(stride)-stride+1).view(1, 1, -1).repeat(num_layers, num_heads, 1).flip(dims=(2,)).to(self.device)), dim=-1)
                elif mode in ['roco']:
                    cur_std = torch.sqrt(cache_attn_scores_square / cache_counter - (cache_attn_scores / cache_counter)**2)
                    cur_std[:, :, -10:] = 1e9
                    cur_std[:, :, :sink_length] = 1e9
                    _, feasible_ids = torch.topk(cur_std, largest=False, k=max(budget-recent_window-sink_length, stride), dim=-1) # (layers, heads, k)
                    argmin_id = torch.topk(cache_attn_scores.gather(dim=-1, index=feasible_ids) / cache_counter.gather(dim=-1, index=feasible_ids), dim=-1, largest=False, k=stride)[1] # (layers, heads)
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
                    evict_id = sink_length
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
        n = 0
        output_ids = []
        cache_attn_scores = cache_attn_scores[:, :, :-(stride-1)]
        cache_attn_scores_square = cache_attn_scores_square[:, :, :-(stride-1)]
        cache_counter = cache_counter[:, :, :-(stride-1)]
        assert cache_attn_scores.shape[-1] == _tmp+1
        while n < max_new_tokens:
            next_token = torch.multinomial(prob_prev_step, num_samples=1)
            output_ids.append(next_token[0, 0].cpu().item())
            next_token_prob = torch.gather(raw_prob_prev_step, -1, next_token) # (bsz, 1)
            n += 1
            if output_ids[-1] in eos_token_ids: break
            outputs = self(input_ids=next_token, 
                            past_key_values=past_key_values,
                            attention_mask=torch.ones(next_token.shape[0], 1+past_key_values[0][0].shape[2], dtype=torch.long, device=self.device),
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
            prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)

            # update accumulated attention scores
            if 'h2o_head' == mode:
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, :]
                    cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
            elif 'roco' == mode:
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, :]
                    attention_map_sq = ((outputs.attentions[l][0, :, 0, :])**2)
                    cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                    cache_attn_scores_square[l, :, :attention_map_sq.shape[-1]] += attention_map_sq
            elif 'tova' == mode:
                for l in range(num_layers):
                    attention_map = outputs.attentions[l][0, :, 0, :]
                    cache_attn_scores[l, :, :attention_map.shape[-1]] = attention_map
            cache_counter += 1.0
            recent_ratio = 0.3
            recent_window = int(budget*recent_ratio)
            if mode in ['h2o_head']:
                eviction_ids = torch.argmin(cache_attn_scores[:, :, :-recent_window], dim=-1)
                _eviction_ids = eviction_ids
                eviction_ids = eviction_ids.cpu().numpy().tolist()
                past_key_values = truncate_kv_cache_silo(past_key_values, eviction_ids)
                _index = torch.arange(cache_attn_scores.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).repeat(num_layers, num_heads, 1)
                mask = (_eviction_ids.unsqueeze(-1)!=_index).view(-1, _index.shape[-1])
                cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
            elif mode in ['roco']:
                cur_std = torch.sqrt(cache_attn_scores_square / cache_counter - (cache_attn_scores / cache_counter)**2)
                cur_std[:, :, -10:] = 1e9
                _, feasible_ids = torch.topk(cur_std, largest=False, k=budget-recent_window, dim=-1) # (layers, heads, k)
                argmin_id = torch.argmin(cache_attn_scores.gather(dim=-1, index=feasible_ids) / cache_counter.gather(dim=-1, index=feasible_ids), dim=-1).unsqueeze(-1) # (layers, heads)
                eviction_ids = feasible_ids.gather(dim=-1, index=argmin_id).squeeze(-1)
                _eviction_ids = eviction_ids
                eviction_ids = eviction_ids.cpu().numpy().tolist()
                past_key_values = truncate_kv_cache_silo(past_key_values, eviction_ids)
                _index = torch.arange(cache_attn_scores.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).repeat(num_layers, num_heads, 1)
                mask = (_eviction_ids.unsqueeze(-1)!=_index).view(-1, _index.shape[-1])
                cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                cache_attn_scores_square = torch.cat((cache_attn_scores_square.view(-1, cache_attn_scores_square.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
                cache_counter = torch.cat((cache_counter.view(-1, cache_counter.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
            elif mode == 'tova':
                eviction_ids = torch.argmin(cache_attn_scores, dim=-1)
                _eviction_ids = eviction_ids
                eviction_ids = eviction_ids.cpu().numpy().tolist()
                past_key_values = truncate_kv_cache_silo(past_key_values, eviction_ids)
                _index = torch.arange(cache_attn_scores.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).repeat(num_layers, num_heads, 1)
                mask = (_eviction_ids.unsqueeze(-1)!=_index).view(-1, _index.shape[-1])
                cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, 1, device=self.device)), dim=-1)
            elif mode == 'recency':
                past_key_values = truncate_kv_cache(past_key_values, start=sink_length, end=sink_length+1)
            elif mode == 'random':
                scores = torch.rand(*positions_tensor.shape).to(self.device)
                _, evict_id = torch.topk(scores, k=1, dim=-1)
                evict_id = evict_id[0].cpu().item()
                past_key_values = truncate_kv_cache(past_key_values, start=sink_length+evict_id, end=sink_length+evict_id+1)
            cur_pos_id += 1
        cache_size = past_key_values[0][0].shape[2]
        total_length = length + len(output_ids)
        print(f"KV Cache Budget ratio {cache_size / total_length*100:.2f}%[{cache_size}/({length}+{len(output_ids)})]")
        decoded_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return decoded_output
    elif kv_mode == 'ppl':
        """
        perplexity computation with fixed kv cache
        """
        length = input_ids.shape[-1]
        if budget >= 1.0:
            outputs_prefilling = self(input_ids=input_ids, use_cache=False)
            logits = outputs_prefilling.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            log_probs = loss_fct(logits[0, :-1], input_ids.clone()[0, 1:]).cpu().numpy().tolist()
            ppl = math.exp(statistics.mean(log_probs))
            return ppl
        else:
            # In case budget is also large, the attention_map will occupy a lot of memory
            # We offload attention_map to CPU first and move it layer by laer to GPU to compute eviction score
            if 'llama' in self.config.architectures[0].lower():
                modify_method_of_instance(self, "LlamaAttention", "forward", partial(llama_forward if not streaming else llama_forward_stream, attn_device='cuda'))
            else:
                modify_method_of_instance(self, "MistralAttention", "forward", partial(mistral_forward if not streaming else mistral_forward_stream, attn_device='cuda'))
            if type(budget) == float:
                budget = int(length * budget) + stride
            elif type(budget) == int: 
                budget += stride
            for idx in range(budget, -1, -1):
                if (length-idx)%stride==0: break
            for r_idx in range(1, idx):
                if (idx-r_idx)%stride==0: break
            prefix = input_ids[:, :r_idx]
            recent_window = int(budget*recent_ratio)
            sink_length = temp_length
            outputs_prefilling = self(input_ids=prefix, use_cache=True, output_attentions=keep_attention)
            past_key_values, logits = outputs_prefilling.past_key_values, outputs_prefilling.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            logits_prev_step = logits[:, -1, :]
            _, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)
            prefix_token_lst = input_ids[0].cpu().numpy().tolist()
            cache_tokens = prefix[0].cpu().numpy().tolist()
            if keep_attention:
                outputs_prefilling.attentions = process_for_mqa_gqa(outputs_prefilling.attentions, num_layers, num_heads, rep_n)
            cache_attn_scores, cache_attn_scores_square = h2o_head_score(outputs_prefilling.attentions, self.device, stride, idx, num_layers, num_heads, empty=not keep_attention)
            # Back to GPU
            if 'llama' in self.config.architectures[0].lower():
                modify_method_of_instance(self, "LlamaAttention", "forward", partial(llama_forward if not streaming else llama_forward_stream, attn_device='cuda'))
            else:
                modify_method_of_instance(self, "MistralAttention", "forward", partial(mistral_forward if not streaming else mistral_forward_stream, attn_device='cuda'))

            if keep_attention:
                cache_counter = torch.tensor([[[1.0]*(idx+stride) for _ in range(num_heads)] for _ in range(num_layers)], device=self.device)
                cache_counter = torch.cumsum(cache_counter, dim=-1).flip(dims=(2,)) - float(stride)
            else:
                cache_counter = torch.tensor([[[float(stride)]*idx+torch.arange(stride, 0, -1).numpy().tolist() for _ in range(num_heads)] for _ in range(num_layers)], device=self.device) - float(stride)
            cache_counter_token = torch.tensor([1.0]*(idx+stride), device=self.device)
            cache_counter_token = torch.cumsum(cache_counter_token, dim=-1).flip(dims=(0, )) - float(stride)
            n = 0
            output_ids = []
            token_probs = []
            cur_pos_id = past_key_values[0][0].shape[2]
            evicted_positions = []
            log_probs = []
            all_logits = []
            all_ids = []
            # for token_i in range(idx, length, stride):
            for token_i in range(r_idx, length, stride):
                n += stride
                outputs = self(input_ids=input_ids[:, token_i:token_i+stride],
                                past_key_values=past_key_values,
                                attention_mask=torch.ones(1, stride+past_key_values[0][0].shape[2], dtype=torch.long, device=self.device),
                                position_ids=torch.LongTensor(list(range(cur_pos_id, cur_pos_id+stride))).to(self.device).view(1, -1),
                                use_cache=True,
                                output_attentions=True)
                past_key_values = outputs.past_key_values
                logits_prev_step = outputs.logits[:, -1, :]
                all_logits.append(outputs.logits[0])
                all_ids.append(input_ids[0, token_i:token_i+stride])
                prob_prev_step, raw_prob_prev_step = logits_adapter(logits_prev_step, temperature, top_p)

                # Unified processing for MQA, GQA and MHA
                outputs.attentions = process_for_mqa_gqa(outputs.attentions, num_layers, num_heads, rep_n)
                cur_kv_size = past_key_values[0][0].shape[2]
                # update accumulated attention scores
                if cur_kv_size>idx or keep_attention:
                    if 'h2o_head' == mode:
                        for l in range(num_layers):
                            attention_map = outputs.attentions[l][0, :, :, :].sum(dim=1) # (num_heads, stride, stride+l)
                            cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                    elif 'roco' == mode:
                        for l in range(num_layers):
                            attention_map = outputs.attentions[l][0, :, :, :].sum(dim=1) # (num_heads, l)
                            attention_map_sq = ((outputs.attentions[l][0, :, :, :])**2).sum(dim=1)
                            cache_attn_scores[l, :, :attention_map.shape[-1]] += attention_map
                            cache_attn_scores_square[l, :, :attention_map.shape[-1]] += attention_map_sq
                    elif 'tova' == mode:
                        for l in range(num_layers):
                            attention_map = outputs.attentions[l][0, :, -1, :].mean(dim=0).unsqueeze(0).repeat(num_heads, 1) # (num_heads, l)
                            cache_attn_scores[l, :, :attention_map.shape[-1]] = attention_map
                # evict if current kv cache size exceeds the budget
                if mode != 'full' and cur_kv_size>idx:
                    cache_counter += float(stride)
                    cache_counter_token += float(stride)
                    if mode in ['h2o_head']:
                        eviction_ids = torch.topk(cache_attn_scores[:, :, sink_length:-recent_window], dim=-1, k=stride, largest=False)[1] + sink_length
                        past_key_values = truncate_kv_cache_liso(past_key_values, eviction_ids)
                        _index = torch.ones(num_layers, num_heads, cache_attn_scores.shape[-1], device=self.device).view(num_layers*num_heads, -1)
                        _src = torch.zeros(num_layers, num_heads, stride, device=self.device).view(num_layers*num_heads, -1)
                        mask = _index.scatter(dim=-1, index=eviction_ids.view(num_layers*num_heads, -1), src=_src).bool()
                        cache_attn_scores = torch.cat((cache_attn_scores.view(-1, cache_attn_scores.shape[-1])[mask].view(num_layers, num_heads, -1), torch.zeros(num_layers, num_heads, stride, device=self.device)), dim=-1)
                        cache_counter = torch.cat((cache_counter.view(-1, cache_counter.shape[-1])[mask].view(num_layers, num_heads, -1), (torch.arange(stride)-stride+1).view(1, 1, -1).repeat(num_layers, num_heads, 1).flip(dims=(2,)).to(self.device)), dim=-1)
                    elif mode in ['roco']:
                        cur_std = torch.sqrt(cache_attn_scores_square / cache_counter - (cache_attn_scores / cache_counter)**2)
                        cur_std[:, :, -10:] = 1e9
                        cur_std[:, :, :sink_length] = 1e9
                        _, feasible_ids = torch.topk(cur_std, largest=False, k=max(budget-recent_window-sink_length, stride), dim=-1) # (layers, heads, k)
                        # _, feasible_ids = torch.topk(cur_std, largest=False, k=max(budget-int(budget*0.1)-sink_length, stride), dim=-1) # (layers, heads, k)
                        argmin_id = torch.topk(cache_attn_scores.gather(dim=-1, index=feasible_ids) / cache_counter.gather(dim=-1, index=feasible_ids), dim=-1, largest=False, k=stride)[1] # (layers, heads)
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
                        evict_id = sink_length
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
            all_ids = torch.cat(all_ids)
            all_logits = torch.cat(all_logits, dim=0)
            assert all_ids.shape[0] == all_logits.shape[0]
            log_probs = loss_fct(all_logits[:-1], all_ids[1:]).cpu().numpy().tolist()
            ppl = math.exp(statistics.mean(log_probs))
            return ppl

def enable_fixed_kv(model, tokenizer, mode, stride=1, verbose=False):
    model.tokenizer = tokenizer
    import functools
    model.easykv_generate = functools.partial(generate, self=model, kv_mode=mode, stride=stride, report_decoding_latency=verbose)
    model.easykv_ppl = functools.partial(generate, self=model, kv_mode='ppl', stride=stride)
    print(f"Fixed KV Cache for {mode} enabled")