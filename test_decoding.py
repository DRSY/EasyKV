import warnings
warnings.filterwarnings("ignore")
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer)
from easykv import enable_fixed_kv

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

model_name = 'llama2_7b_chat'
path = MODEL_CONFIGS[model_name]['path']
template = MODEL_CONFIGS[model_name]['template']
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map='auto').eval()
tokenizer = AutoTokenizer.from_pretrained(path)

# Setup KV caching mode
enable_fixed_kv(model, tokenizer, mode='decoding', stride=1)

# Test input
inst = "What are the names of some famous actors that started their careers on Broadway?"
input_prompt = template.format(inst=inst)

# Define eviction policy
kv_policy = 'roco'

for budget in [300, 150]:
    # Define sampling parameters
    gen_kwargs = dict(
        temperature=1e-9,
        top_p=1.0,
        max_new_tokens=2048,
        budget=budget,
        kv_policy=kv_policy,
    )
    input_ids = tokenizer([input_prompt], return_tensors='pt').input_ids.to(model.device)
    output = model.easykv_generate(input_ids=input_ids, generation_config=gen_kwargs)
    print(f"{'='*20} EasyKV-{kv_policy}-{budget} {'='*20}\n{output}")