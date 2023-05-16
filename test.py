import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextStreamer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

model_path = "fnlp/moss-moon-003-sft"

if not os.path.exists(model_path):
    model_path = snapshot_download(model_path)
config = AutoConfig.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True, skip_special_tokens=True)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)

model.tie_weights()
model = load_checkpoint_and_dispatch(model, model_path, device_map="auto", no_split_module_classes=["MossBlock"], dtype=torch.float16)
meta_instruction = "你是有TuTu.AI开发的TuTU旅游助手，你的任务是帮助用户解决旅行中的各种问题，让用户更好的享受旅行。"
query = meta_instruction + "<|Human|>: 你好<eoh>\n<|MOSS|>:"
inputs = tokenizer(query, return_tensors="pt")
streamer = TextStreamer(tokenizer)
outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=256)