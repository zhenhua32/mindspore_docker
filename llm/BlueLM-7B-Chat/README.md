---
license: other
language:
- zh
- en
---
# BlueLM

<p align="center">
🖥 <a href="https://github.com/vivo-ai-lab/BlueLM" target="_blank">github</a>  • 📜 <a href="https://huggingface.co/vivo-ai/BlueLM-7B-Chat/blob/main/MODEL_LICENSE" target="_blank">LICENSE</a> • 🎯 <a href="https://developers.vivo.com/product/ai/bluelm" target="_blank">vivo Developers</a> • 🗨 <a href="https://github.com/vivo-ai-lab/BlueLM/blob/main/resources/wechat.png" target="_blank">WeChat</a>
</p>

## 模型介绍/Introduction

BlueLM 是由 vivo AI 全球研究院自主研发的大规模预训练语言模型，本次发布包含 7B 基础模型和 7B 对话模型，同时我们开源了支持 **32K** 的长文本基础模型和对话模型。

- **更大量的优质数据**：高质量语料库进行训练，规模达到了 **2.6 万亿** 的 token 数，该语料库包含中文、英文以及少量日韩数据。
- **更优的效果**：其中 BlueLM-7B-Chat 在 **C-Eval** 和 **CMMLU** 上均取得领先结果，对比同尺寸开源模型中具有较强的竞争力。
- **长文本支持**：BlueLM-7B-Base-32K 和 BlueLM-7B-Chat-32K 均支持 **32K** 长文本，在保持基础能力相当情况下，能够支持更长上下文理解。
- **协议说明**：BlueLM 系列欢迎开发者进行学术研究和商业应用。

BlueLM is a large-scale open-source language model independently developed by the vivo AI Lab. This release includes 2K and 32K context length versions for both Base and Chat models.

- **High-quality Data**: BlueLM is trained on a high-quality data with 2.6 trillion tokens. Our train corpus mainly consists of Chinese and English data, with a small amount of Japanese and Korean data.
- **Stronger Performance**: BlueLM-7B-Chat achieves a strong competitive performance in C-Eval and CMMLU benchmarks of the same size.
- **Longer Context**: We have extended the context length of both BlueLM-7B-Base-32K and BlueLM-7B-Chat-32K models from 2K to 32K. The models can support longer context understanding while maintaining the same basic capabilities.
- **Model License**: BlueLM weights are open for academic research and commercial use. 

本次发布基座模型下载链接见：

The release versions and hugging face download links are listed in the table below:

|     |          Base Model        |          Chat Model        |       4bits Quantized Chat Model        |
|:---:|:--------------------:|:--------------------:|:--------------------------:|
| 7B-2k  | [BlueLM-7B-Base](https://huggingface.co/vivo-ai/BlueLM-7B-Base)  | [BlueLM-7B-Chat](https://huggingface.co/vivo-ai/BlueLM-7B-Chat)  | [BlueLM-7B-Chat-4bits](https://huggingface.co/vivo-ai/BlueLM-7B-Chat-4bits)  |
| 7B-32K | [BlueLM-7B-Base-32K](https://huggingface.co/vivo-ai/BlueLM-7B-Base-32K) | [BlueLM-7B-Chat-32K](https://huggingface.co/vivo-ai/BlueLM-7B-Chat-32K) | - |

## 评测结果/Benchmark Results

为了保证模型评测的一致性，我们采用 [OpenCompass](https://opencompass.org.cn/leaderboard-llm) 进行相关榜单的评测。我们分别在 C-Eval、MMLU、CMMLU、GaoKao、AGIEval、BBH、GSM8K、MATH 和 HumanEval 榜单对 BlueLM 的通用能力、数学能力和代码能力进行了测试。

To ensure the consistency of model evaluation, we use [OpenCompass](https://opencompass.org.cn/leaderboard-llm) to evaluate the performance on relevant leaderboards. We conducted extensive tests on C-Eval, MMLU, CMMLU, GaoKao, AGIEval, BBH, GSM8K, MATH and HumanEval datasets across general ability, mathematical ability and coding ability.

| Model             | **C-Eval** | **MMLU** | **CMMLU** | **Gaokao** | **AGIEval** | **BBH** | **GSM8K** | **MATH** | **HumanEval** |
|:------------------|:-----------|:---------|:----------|:-----------|:------------|:--------|:----------|:---------|:--------------|
|                   | 5-shot     | 5-shot   | 5-shot    | 0-shot     | 0-shot      | 3-shot  | 4-shot    | 5-shot   | 0-shot        |
| GPT-4             | 69.9       | 86.4     | 71.2      | 72.3       | 55.1        | 86.7    | 91.4      | 45.8     | 74.4          |
| ChatGPT           | 52.5       | 70.0     | 53.9      | 51.1       | 39.9        | 70.1    | 78.2      | 28       | 73.2          |
| LLaMA2-7B         | 32.5       | 45.3     | 31.8      | 18.9       | 21.8        | 38.2    | 16.7      | 3.3      | 12.8          |
| ChatGLM2-6B(Base) | 51.7       | 47.9     | 50.0      | -          | -           | 33.7    | 32.4      | 6.5      | -             |
| Baichuan2-7B      | 56.3       | 54.7     | 57.0      | 34.8       | 34.6        | 41.8    | 24.6      | 5.4      | 17.7          |
| BlueLM-7B-Base    | 67.5       | 55.2     | 66.6      | 58.9       | 43.4        | 41.7    | 27.2      | 6.2      | 18.3          |
| BlueLM-7B-Chat    | 72.7       | 50.7     | 74.2      | 48.7       | 43.4        | 65.6    | 51.9      | 13.4     | 21.3          |

## 推理部署/Inference and Deployment

```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("vivo-ai/BlueLM-7B-Chat", trust_remote_code=True, use_fast=False)
>>> model = AutoModelForCausalLM.from_pretrained("vivo-ai/BlueLM-7B-Chat", device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True)
>>> model = model.eval()
>>> inputs = tokenizer("[|Human|]:三国演义的作者是谁？[|AI|]:", return_tensors="pt")
>>> inputs = inputs.to("cuda:0")
>>> pred = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
>>> print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
三国演义的作者是谁？ 《三国演义》是元末明初小说家罗贯中创作的长篇小说。
```

更多使用说明，请参考我们的 [Github 仓库](https://github.com/vivo-ai-lab/BlueLM)。

For more instructions, please refer to our [Github Repo](https://github.com/vivo-ai-lab/BlueLM).

## 协议/License

社区使用代码依照 [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) 协议开源，且使用 BlueLM 模型权重需要遵循 [vivo_BlueLM模型许可协议](https://huggingface.co/vivo-ai/BlueLM-7B-Chat/blob/main/MODEL_LICENSE)。

Our code is licensed under the [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) and [Community License for BlueLM Model](https://huggingface.co/vivo-ai/BlueLM-7B-Chat/blob/main/MODEL_LICENSE).