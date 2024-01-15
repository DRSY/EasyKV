<h1 align="center">
<img src="./logo.png" alt="EasyKV" width="250" height="200"/>
<br>
EasyKV
</h1>

EasyKV is a Pytorch implementation of various eviction policies for ***key-value cache constrained*** generative language model inference.
<p align="center">
  <a href="#update">Update</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#example-usage">Example</a> •
  <a href="#passkey-retrieval-example">Passkey Retrieval</a> •
  <a href="#summarization-example">Summarization</a> •
  <a href="#instruction-following">Instruction Following</a> •
  <a href="#acknowledgement">Acknowledgement</a>
</p>

## Update
+ [2024.1.16] Add examples for [Instruction Following](#instruction-following) using LLaMa2-7B-Chat.
+ [2024.1.15] Add examples for [Passkey Retrieval](#passkey-retrieval-example) using long-context LLM(Vicuna-7B-16K) and DynamicNTK-scaled LLaMa2-7B-Chat.
+ [2024.1.15] Add examples for [Summarization](#summarization-example) using LLaMa2-7B-Chat.
+ [2024.1.14] Uploaded the standalone Pytorch implementation. Pypi package and paper describing the details of our integrated eviction policy design are coming soon.

## Features
+ Offer control over the memory budget allocated for the KV cache during LLM inference, with easy-to-use interface.
+ Support both prompt encoding and auto-regressive decoding.
+ Support Multi-Head Attention(MHA), Multi-Query Attention(MQA), and Grouped-Query Attention(GQA).
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
from easykv import enable_fixed_kv
```

## Example Usage
There are two different phases in LLM generative inference, i.e., prompt encoding and auto-regressive decoding. Firstly, some necessary patches to ```transformers``` are required (minor changes to RoPE to make it compatible with asynchronous position_ids and key-value length).
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from easykv import enable_fixed_kv
``` 
### Prompt Encoding/Prefilling
For prefilling stage, please specify ```budget``` in the range of (0,1), e.g., 0.5, which leads to 50% savings in KV cache memory footprint.
```python

# Define your model path and template in a dict MODEL_CONFIGS
model_name = 'zephyr_7b'
path = MODEL_CONFIGS[model_name]['path']
template = MODEL_CONFIGS[model_name]['template']
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map='auto').eval()
tokenizer = AutoTokenizer.from_pretrained(path)

# Turn on fixed KV cache mode for prefilling phase
stride=8
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
    kv_policy=kv_policy,
    keep_attention=False # set to True if your DRAM is not tight and you can get better performance
)
input_ids = tokenizer([input_prompt], return_tensors='pt').input_ids.to(model.device)
output = model.easykv_generate(input_ids=input_ids, generation_config=gen_kwargs)
print(f"{'='*20} {kv_policy} {'='*20}\n{output}")
```
### Auto-regressive Decoding
For auto-regressive decoding phase, please specify ```budget``` as an integer, which represents the maximum length of KV cache, e.g, 200.
```python
# Define your model path and template in a dict MODEL_CONFIGS
model_name = 'llama2_7b_chat'
path = MODEL_CONFIGS[model_name]['path']
template = MODEL_CONFIGS[model_name]['template']
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map='auto').eval()
tokenizer = AutoTokenizer.from_pretrained(path)

# Turn on fixed KV cache mode for decoding phase
enable_fixed_kv(model, tokenizer, mode='decoding', stride=1)

# Test input
prompt = f"What are the names of some famous actors that started their careers on Broadway?"
input_prompt = template.format(inst=prompt)
kv_policy = 'h2o_head_decay_avg_std'
# Define sampling parameters
gen_kwargs = dict(
    temperature=1e-9,
    top_p=1.0,
    max_new_tokens=2048,
    budget=200,
    kv_policy=kv_policy,
)
input_ids = tokenizer([input_prompt], return_tensors='pt').input_ids.to(model.device)
output = model.easykv_generate(input_ids=input_ids, generation_config=gen_kwargs)
print(f"{'='*20} {kv_policy} {'='*20}\n{output}")
```

### Passkey Retrieval Example
We provide examplar code for passkey retrieval in [test_passkey.py](./test_passkey.py) and [test_passkey_NTK.py](./test_passkey_NTK.py) using Vicuna-7B-16K and DynamicNTK-scaled LLaMa2-7B-Chat, respectively.

The results of DynamicNTK-scaled LLaMa2-7B-Chat on ```5K``` passkey retrieval task is shown below:
```bash
#Tokens of Prompt: 5144 Passkey target: 89427
KV cache budget ratio: 100.00%(5144/5144)
Current GPU memory usage: 18.359 GB
Peak GPU memory usage: 21.751 GB
Llama2-EasyKV-h2o_head_std_avg(100.00%):     [What is the pass key? The pass key is 89427.]

KV cache budget ratio: 50.08%(2576/5144)
Current GPU memory usage: 15.625 GB
Peak GPU memory usage: 18.423 GB
Llama2-EasyKV-h2o_head_std_avg(50.00%):     [What is the pass key? The pass key is 89427.]
------------------------------------------------------------------------------------
#Tokens of Prompt: 5144 Passkey target: 51906
KV cache budget ratio: 100.00%(5144/5144)
Current GPU memory usage: 18.359 GB
Peak GPU memory usage: 21.751 GB
Llama2-EasyKV-h2o_head_std_avg(100.00%):     [What is the pass key? The pass key is 51906.]

KV cache budget ratio: 50.08%(2576/5144)
Current GPU memory usage: 15.625 GB
Peak GPU memory usage: 18.427 GB
Llama2-EasyKV-h2o_head_std_avg(50.00%):     [What is the pass key? The pass key is 51906.]
------------------------------------------------------------------------------------
#Tokens of Prompt: 5144 Passkey target: 38117
KV cache budget ratio: 100.00%(5144/5144)
Current GPU memory usage: 18.359 GB
Peak GPU memory usage: 21.751 GB
Llama2-EasyKV-h2o_head_std_avg(100.00%):     [What is the pass key? The pass key is 38117.]

KV cache budget ratio: 50.08%(2576/5144)
Current GPU memory usage: 15.625 GB
Peak GPU memory usage: 18.427 GB
Llama2-EasyKV-h2o_head_std_avg(50.00%):     [What is the pass key? The pass key is 38117.]
------------------------------------------------------------------------------------
#Tokens of Prompt: 5144 Passkey target: 60151
KV cache budget ratio: 100.00%(5144/5144)
Current GPU memory usage: 18.359 GB
Peak GPU memory usage: 21.751 GB
Llama2-EasyKV-h2o_head_std_avg(100.00%):     [What is the pass key? The pass key is 60151.]

KV cache budget ratio: 50.08%(2576/5144)
Current GPU memory usage: 15.625 GB
Peak GPU memory usage: 18.427 GB
Llama2-EasyKV-h2o_head_std_avg(50.00%):     [What is the pass key? The pass key is 60151.]
------------------------------------------------------------------------------------
#Tokens of Prompt: 5144 Passkey target: 23789
KV cache budget ratio: 100.00%(5144/5144)
Current GPU memory usage: 18.359 GB
Peak GPU memory usage: 21.752 GB
Llama2-EasyKV-h2o_head_std_avg(100.00%):     [What is the pass key? The pass key is 23789.]

KV cache budget ratio: 50.08%(2576/5144)
Current GPU memory usage: 15.626 GB
Peak GPU memory usage: 18.427 GB
Llama2-EasyKV-h2o_head_std_avg(50.00%):     [What is the pass key? The pass key is 23789.]
```

The results of Vicuna-7B-16K on ```10K``` passkey retrieval task is shown below:
```bash
#Tokens of Prompt: 9994 Passkey target: 51013
KV cache budget ratio: 100.00%(9994/9994)
Current GPU memory usage: 23.666 GB
Peak GPU memory usage: 41.896 GB
EasyKV-h2o_head_std_avg(100.00%):     [What is the pass key? The pass key is 51013.]

KV cache budget ratio: 50.05%(5002/9994)
Current GPU memory usage: 18.4 GB
Peak GPU memory usage: 25.36 GB
EasyKV-h2o_head_std_avg(50.00%):     [What is the pass key? The pass key is 51013.]
------------------------------------------------------------------------------------
#Tokens of Prompt: 9994 Passkey target: 36920
KV cache budget ratio: 100.00%(9994/9994)
Current GPU memory usage: 23.666 GB
Peak GPU memory usage: 41.896 GB
EasyKV-h2o_head_std_avg(100.00%):     [What is the pass key? The pass key is 36920.]

KV cache budget ratio: 50.05%(5002/9994)
Current GPU memory usage: 18.378 GB
Peak GPU memory usage: 25.36 GB
EasyKV-h2o_head_std_avg(50.00%):     [What is the pass key? The pass key is 36920.]
------------------------------------------------------------------------------------
#Tokens of Prompt: 9994 Passkey target: 83493
KV cache budget ratio: 100.00%(9994/9994)
Current GPU memory usage: 23.666 GB
Peak GPU memory usage: 41.896 GB
EasyKV-h2o_head_std_avg(100.00%):     [What is the pass key? The pass key is 83493.]

KV cache budget ratio: 50.05%(5002/9994)
Current GPU memory usage: 18.378 GB
Peak GPU memory usage: 25.36 GB
EasyKV-h2o_head_std_avg(50.00%):     [What is the pass key? The pass key is 83493.]
------------------------------------------------------------------------------------
#Tokens of Prompt: 9994 Passkey target: 78585
KV cache budget ratio: 100.00%(9994/9994)
Current GPU memory usage: 23.666 GB
Peak GPU memory usage: 41.896 GB
EasyKV-h2o_head_std_avg(100.00%):     [What is the pass key? The pass key is 78585.]

KV cache budget ratio: 50.05%(5002/9994)
Current GPU memory usage: 18.378 GB
Peak GPU memory usage: 25.36 GB
EasyKV-h2o_head_std_avg(50.00%):     [What is the pass key? The pass key is 78585.]
------------------------------------------------------------------------------------
#Tokens of Prompt: 9994 Passkey target: 58328
KV cache budget ratio: 100.00%(9994/9994)
Current GPU memory usage: 23.666 GB
Peak GPU memory usage: 41.896 GB
EasyKV-h2o_head_std_avg(100.00%):     [What is the pass key? The pass key is 58328.]

KV cache budget ratio: 50.05%(5002/9994)
Current GPU memory usage: 18.378 GB
Peak GPU memory usage: 25.36 GB
EasyKV-h2o_head_std_avg(50.00%):     [What is the pass key? The pass key is 58328.]
```

### Summarization Example
We provide examplar code for summarization in [test_summarization.py](./test_summarization.py).
The results of full KV cache and 50%-constrained KV cache using EasyKV is shown below:
```bash
EasyKV(100.00%): The 2016 European Championship, also known as Euro 2016, will take place in France from June 10 to July 10, featuring 24 teams, including France, Spain, Germany, England, Wales, and Northern Ireland, with the tournament kicking off with France playing Romania on Friday, June 10, and the final taking place at the Stade de France in Paris on July 10.
EasyKV(50.00%): The 2016 European Championship, also known as Euro 2016, will be held in France from June 10th to July 10th, featuring 24 teams, including defending champions Spain, and will be marked by a number of changes to the rules of the game, as well as increased security measures due to the ongoing terror threat.
```

### Instruction Following
We provide examplar code for instruction-following in [test_decoding.py](./test_decoding.py).
The results of EasyKV using different KV budget(300/150) with LLaMa2-7B-Chat are shown below:
```bash
Instruction: What are the names of some famous actors that started their careers on Broadway?
KV cache budget ratio: 58.14%(300/516)
==================== EasyKV-h2o_head_decay_avg_std-300 ====================
Thank you for your question! There are many talented actors who have started their careers on Broadway and gone on to achieve great success in the entertainment industry. Here are some famous actors who got their start on Broadway:
1. Hugh Jackman: Known for his roles in "The Boy from Oz" and "The Greatest Showman," Jackman got his start on Broadway in the musical "Oklahoma!"
2. Audra McDonald: A six-time Tony Award winner, McDonald got her start on Broadway in the musical "Ragtime."
3. Idina Menzel: Menzel got her start on Broadway in the musical "Rent" and went on to star in "Wicked" and "Frozen."
4. Lin-Manuel Miranda: Miranda got his start on Broadway in the musical "In the Heights" and went on to create the hit musical "Hamilton."
5. Bernadette Peters: A legendary actress and singer, Peters got her start on Broadway in the musical "The Beautyful Nose" and has since starred in numerous productions, including "Gypsy" and "Sweeney Todd."
6. James Corden: Corden got his start on Broadway in the musical "Les Miserables" before becoming a late-night talk show host on "The Late Late Show with James Corden."
7. Christine Baranski: Baranski got her start on Broadway in the musical "The Producers" and has since appeared in numerous productions, including "The Good Wife" and "The Good Fight."
8. Nathan Lane: Lane got his start on Broadway in the musical "A Funny Thing Happened on the Way to the Forum" and has since starred in numerous productions, including "The Producers" and "It's Only a Play."
9. Bette Midler: Midler got her start on Broadway in the musical "Fiddler on the Roof" and went on to star in "Wicked" and "Hello, Dolly!"
10. John Leguizamo: Leguizamo got his start on Broadway in the play "A Day in the Death of Jose Marti" and has since appeared in numerous productions, including "Spanglish" and "The Lion King."
I hope this list helps! Let me know if you have any other questions.
KV cache budget ratio: 28.30%(150/530)
==================== EasyKV-h2o_head_decay_avg_std-150 ====================
Thank you for your question! There are many talented actors who have started their careers on Broadway and gone on to achieve great success in the entertainment industry. Here are some famous actors who got their start on Broadway:
1. Hugh Jackman: Known for his roles in "The Boy from Oz" and "The Greatest Showman," Jackman got his start on Broadway in the musical "Oklahoma!"
2. Audra McDonald: A six-time Tony Award winner, McDonald got her start on Broadway in the musical "Ragtime."
3. Idina Menzel: Menzel got her start on Broadway in the musical "Rent" and went on to star in "Wicked" and "Frozen."
4. Lin-Manuel Miranda: Miranda got his start on Broadway in the musical "In the Heights" and went on to create the hit musical "Hamilton."
5. Bernadette Peters: A legendary actress and singer, Peters got her start on Broadway in the musical "The Beautyful Nose" and has since starred in numerous Broadway productions.
6. James Corden: Corden got his start on Broadway in the musical "Les Miserables" before becoming a late-night talk show host on "The Late Late Show with James Corden."
7. Christine Baranski: Baranski got her start on Broadway in the musical "The Producers" before going on to star in the TV show "The Good Wife" and the movie "The Big Sick."
8. Nathan Lane: Lane got his start on Broadway in the musical "A Funny Thing Happened on the Way to the Forum" and has since starred in numerous Broadway productions, including "The Producers" and "The Birdcage."
9. Bette Midler: Midler got her start on Broadway in the musical "Fiddler on the Roof" before going on to star in the TV show "The Rose" and the movie "Hocus Pocus."
10. John Leguizamo: Leguizamo got his start on Broadway in the play "A Day in the Death of Jose Marti" before going on to star in numerous TV shows and movies, including "ER" and "Ice Age."
These are just a few examples of actors who got their start on Broadway. There are many other talented actors who have also gotten their start on the Great White Way.
```

## List of Supported KV Eviction Policies:
+ random: drop kv cache of a randomly chosen position.
+ recency: similar to StreamingLLM, dropping the least recent token's kv cache.
+ h2o_head: Heavy-hitter oracle, which drops kv cache whose accumulated attention score is smallest.
+ tova: Token Omission Via Attention, which uses attention weights of the last token only.
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

@article{liu2023scissorhands,
  title={Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time},
  author={Liu, Zichang and Desai, Aditya and Liao, Fangshuo and Wang, Weitao and Xie, Victor and Xu, Zhaozhuo and Kyrillidis, Anastasios and Shrivastava, Anshumali},
  journal={arXiv preprint arXiv:2305.17118},
  year={2023}
}

@article{zhang2023h,
  title={H $ \_2 $ O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models},
  author={Zhang, Zhenyu and Sheng, Ying and Zhou, Tianyi and Chen, Tianlong and Zheng, Lianmin and Cai, Ruisi and Song, Zhao and Tian, Yuandong and R{\'e}, Christopher and Barrett, Clark and others},
  journal={arXiv preprint arXiv:2306.14048},
  year={2023}
}
```