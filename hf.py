import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MODEL_NAME = "Qwen/Qwen-VL-Chat"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    # trust_remote_code=True,
    cache_dir=TOKEN_CACHE
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    cache_dir=MODEL_CACHE
).to("cuda")

model.generation_config = GenerationConfig.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

query = tokenizer.from_list_format([
    {'image': 'poster.jpeg'},
    {'text': 'What is the name of the movie in the poster?'},
])

response, history = model.chat(tokenizer, query=query, history=None)
print(response)