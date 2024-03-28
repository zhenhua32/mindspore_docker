# 模型下载
from modelscope import snapshot_download

# model_dir = snapshot_download("qwen/Qwen-7B-Chat", cache_dir="./model")
model_dir = snapshot_download("deepseek-ai/deepseek-llm-7b-chat", cache_dir="./model")
