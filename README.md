# EasyKV
EasyKV is a Pytorch implementation of various eviction policies for key-value cache constrained generative language model inference.

## Update
+ Uploaded the standalone Pytorch implementation. Pypi package and paper describing the details of our integrated eviction policy design are coming soon.

## Features
+ Offer control over the memory budget allocated for the KV cache during LLM inference, with easy-to-use interface.
+ Support both prompt encoding and auto-regressive decoding.
+ Support Multi-head Attention, Multi-query Attention, and Qrouped-query Attention.
+ Support LLaMa, LLaMa2, and Mistral.
+ Support various stride for prompt encoding(larger stride leads to faster encoding).

## Installation
First of all, clone this repo into your working directory.
```bash
git clone https://github.com/DRSY/EasyKV.git
cd EasyKV
```
Then import ```enable_fixed_kv``` in your Python script:
```python
from kv_utils import enable_fixed_kv
```

## Example Usage
For prefilling stage:
```python
import transformers
transformers.models.llama.modeling_llama.LlamaAttention.forward = modeling_llama.llama_forward
transformers.models.mistral.modeling_mistral.MistralAttention.forward = modeling_mistral.mistral_forward
from transformers import AutoModelForCausalLM, AutoTokenizer

# define your model path and template in a dict MODEL_CONFIGS
model_name = 'zephyr_7b'
path = MODEL_CONFIGS[model_name]['path']
template = MODEL_CONFIGS[model_name]['template']
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map='auto').eval()
tokenizer = AutoTokenizer.from_pretrained(path)

from kv_utils import enable_fixed_kv
# for prompt encoding, we set mode to 'encoding'
stride=5
enable_fixed_kv(model, tokenizer, mode='encoding', stride=stride)
# Test input
article = "###\nArticle: It was the first time the Single Transferable Vote (STV) system had been used to select two members in the same ward in a by-election. The SNP topped the vote in the Leith Walk by-election, while Scottish Labour won the second seat from the Greens. The by-election was called after Deidre Brock of the SNP and Maggie Chapman of the Scottish Greens stood down. The SNP's John Lewis Ritchie topped the Leith Walk poll with 2,290 votes. He was elected at stage one in the STV process with a swing in first-preference votes of 7.6% from Labour. Labour's Marion Donaldson received 1,623 votes, ahead of Susan Jane Rae of the Scottish Greens on 1,381. Ms Donaldson was elected at stage 10 of the voting process after other preferences had been considered. The by-election was called after Ms Brock stood down when she was elected as the SNP MP for Edinburgh North and Leith in May. Ms Chapman, of the Scottish Greens, resigned from her post to concentrate on standing for the Scottish Parliament in next May's election. The turnout for the by-election was 25.1%. The SNP also held the Midlothian West seat on Midlothian Council with a swing of 6.3% from Labour. The party's Kelly Parry secured 1,540 votes, ahead of Labour's Ian Miller on 945 votes. The by-election was called after Owen Thompson was elected as SNP MP for the Midlothian constituency.\n\nSummarize the above article in 1 sentence.\n"
prompt = f"Write a SHORT summary of the following text delimited by triple backticks. Return your response which covers the key points of the text.\n```{article}```"
input_prompt = template.format(inst=prompt)

# Define eviction policy
kv_policy = 'h2o_head_std_avg'
# Define sampling parameters
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
```
For auto-regressive decoding phase:
```python
import transformers
transformers.models.llama.modeling_llama.LlamaAttention.forward = modeling_llama.llama_forward
transformers.models.mistral.modeling_mistral.MistralAttention.forward = modeling_mistral.mistral_forward
from transformers import AutoModelForCausalLM, AutoTokenizer

# define your model path and template in a dict MODEL_CONFIGS
model_name = 'zephyr_7b'
path = MODEL_CONFIGS[model_name]['path']
template = MODEL_CONFIGS[model_name]['template']
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map='auto').eval()
tokenizer = AutoTokenizer.from_pretrained(path)

from kv_utils import enable_fixed_kv
enable_fixed_kv(model, tokenizer, mode='decoding', stride=1)
prompt = f"What are the names of some famous actors that started their careers on Broadway?"
input_prompt = template.format(inst=prompt)
kv_policy = 'h2o_head_decay_avg_std'
gen_kwargs = dict(
    temperature=1e-9,
    top_p=1.0,
    max_new_tokens=2048,
    budget=200,
    kv_policy=kv_policy
)
input_ids = tokenizer([input_prompt], return_tensors='pt').input_ids.to(model.device)
output = model.generate(input_ids=input_ids, generation_config=gen_kwargs)
print(f"{'='*20} {kv_policy} {'='*20}\n{output}")
```
## List of supported KV Eviction Policies:
+ random: drop kv cache of a randomly chosen position
+ recency: similar to StreamingLLM, dropping the least recent token's kv cache
+ h2o_head: Heavy-hitter oracle, which drops kv cache whose accumulated attention score is smallest
+ h2o_head_std_avg(for encoding mode only): newly proposed eviction policy with better evivtion candidate selection and importance estimation.
+ h2o_head_decay_avg_std(for decoding mode only): newly proposed eviction policy with better evivtion candidate selection and importance estimation.


## Acknowledgement
```latex
@article{xiao2023efficient,
  title={Efficient streaming language models with attention sinks},
  author={Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
  journal={arXiv preprint arXiv:2309.17453},
  year={2023}
}

@article{zhang2023h,
  title={H $ \_2 $ O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models},
  author={Zhang, Zhenyu and Sheng, Ying and Zhou, Tianyi and Chen, Tianlong and Zheng, Lianmin and Cai, Ruisi and Song, Zhao and Tian, Yuandong and R{\'e}, Christopher and Barrett, Clark and others},
  journal={arXiv preprint arXiv:2306.14048},
  year={2023}
}
```