# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generation support."""

from typing import Tuple, List, Union, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from transformers import logging
from transformers.generation import LogitsProcessor

logger = logging.get_logger(__name__)

# Types.
HistoryType = List[Tuple[str, str]]
TokensType = List[int]
BatchTokensType = List[List[int]]


def pad_batch(batch: BatchTokensType, pad_id: int, seq_length: int) -> BatchTokensType:
    for tokens in batch:
        context_length = len(tokens)
        if context_length < seq_length:
            tokens.extend([pad_id] * (seq_length - context_length))
    return batch


def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    reset_position_ids,
    reset_attention_mask,
    eod_mask_loss,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
    ).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def get_batch(context_tokens: torch.LongTensor, eod_id: int):
    """Generate batch from context tokens."""
    # Move to GPU.
    tokens = context_tokens.contiguous().to(context_tokens.device)
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        eod_id,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
    )
    return tokens, attention_mask, position_ids


def get_stop_words_ids(chat_format: str, tokenizer):
    """
    获取多个 停止符, 格式是列表的列表
    """
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        # 这种格式是多轮对话
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    """
    这里就是 qwen 拼接多轮对话的地方
    """
    if history is None:
        history = []

    if chat_format == "chatml":
        # 多轮对话的格式
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            """角色\n对话"""
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        # 系统消息的前后要加上 im_start 和 im_end
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        # 注意是倒序的, 因为要从最新的消息开始拼接, 会截断超过长度的消息, 放弃最老的那些消息
        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            # 用户消息的前后要加上 im_start 和 im_end
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            # 模型消息的前后要加上 im_start 和 im_end
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            # 提问和回答之间都是需要换行符的
            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            # 计算当前的 qa 对能不能加入到 context 中
            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        # 拼接系统消息和对话历史
        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        # 现在要加上最新的用户提问了
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            # 需要加上模型的开始标志, 来引导模型生成
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        # 普通模式, 直接编码就行
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


def _decode_default(
    tokens: List[int],
    *,
    stop_words: List[str],
    eod_words: List[str],
    tokenizer: PreTrainedTokenizer,
    raw_text_len: int,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str='replace',
):
    """解码

    Args:
        tokens (List[int]): 需要解码的 token 列表
        stop_words (List[str]): 不想要的字符串的列表
        eod_words (List[str]): 停止字符列表
        tokenizer (PreTrainedTokenizer): 分词器
        raw_text_len (int): 原始文本的长度
        verbose (bool, optional): 更多信息. Defaults to False.
        return_end_reason (bool, optional): 返回结束原因. Defaults to False.
        errors (str, optional): 错误处理. Defaults to 'replace'.

    Returns:
        _type_: _description_
    """
    # 首先解码, 并去掉原始的文本
    trim_decode_tokens = tokenizer.decode(tokens, errors=errors)[raw_text_len:]
    if verbose:
        print("\nRaw Generate: ", trim_decode_tokens)

    end_reason = f"Gen length {len(tokens)}"
    # 如果有不想要的字符, 就直接去掉
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    # 遇到停止符, 会遍历所有的停止符, 然后不断截断. 这个策略是对的, 当你有所有的 tokens 时
    for eod_word in eod_words:
        if eod_word in trim_decode_tokens:
            end_reason = f"Gen {eod_word!r}"
        trim_decode_tokens = trim_decode_tokens.split(eod_word)[0]
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print("\nEnd Reason:", end_reason)
        print("\nGenerate: ", trim_decode_tokens)

    # 返回结束原因
    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens


def _decode_chatml(
    tokens: List[int],
    *,
    stop_words: List[str],
    eod_token_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    raw_text_len: int,
    context_length: int,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str='replace'
):
    """解码多轮对话形式的

    Args:
        tokens (List[int]): 需要解码的 token 列表
        stop_words (List[str]): 不想要的字符串的列表
        eod_token_ids (List[int]): 停止 token 的列表
        tokenizer (PreTrainedTokenizer): 分词器
        raw_text_len (int): 原始文本的长度
        context_length (int): 上下文长度
        verbose (bool, optional): 更多信息. Defaults to False.
        return_end_reason (bool, optional): 返回结束原因. Defaults to False.
        errors (str, optional): 错误处理. Defaults to 'replace'.

    Returns:
        _type_: _description_
    """
    end_reason = f"Gen length {len(tokens)}"
    eod_token_idx = context_length
    # 从上下文长度开始遍历, 直到遇到停止字符, 获取第一个停止字符的位置
    for eod_token_idx in range(context_length, len(tokens)):
        if tokens[eod_token_idx] in eod_token_ids:
            end_reason = f"Gen {tokenizer.decode([tokens[eod_token_idx]])!r}"
            break

    # 解码, 并去掉原始的文本
    trim_decode_tokens = tokenizer.decode(tokens[:eod_token_idx], errors=errors)[raw_text_len:]
    if verbose:
        print("\nRaw Generate w/o EOD:", tokenizer.decode(tokens, errors=errors)[raw_text_len:])
        print("\nRaw Generate:", trim_decode_tokens)
        print("\nEnd Reason:", end_reason)
    # 将不想要的字符串去掉
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print("\nGenerate:", trim_decode_tokens)

    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens


def decode_tokens(
    tokens: Union[torch.LongTensor, TokensType],
    tokenizer: PreTrainedTokenizer,
    raw_text_len: int,
    context_length: int,
    chat_format: str,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str="replace",
) -> str:
    """解码

    Args:
        tokens (Union[torch.LongTensor, TokensType]): _description_
        tokenizer (PreTrainedTokenizer): _description_
        raw_text_len (int): 原始文本的长度
        context_length (int): 原始 token 的长度
        chat_format (str): _description_
        verbose (bool, optional): _description_. Defaults to False.
        return_end_reason (bool, optional): _description_. Defaults to False.
        errors (str, optional): _description_. Defaults to "replace".

    Raises:
        NotImplementedError: _description_

    Returns:
        str: _description_
    """
    # 转换成列表
    if torch.is_tensor(tokens):
        tokens = tokens.cpu().numpy().tolist()

    if chat_format == "chatml":
        return _decode_chatml(
            tokens,
            stop_words=[],  # 这里是空的
            eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id],  # 有两个停止符
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            context_length=context_length,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    elif chat_format == "raw":
        return _decode_default(
            tokens,
            stop_words=["<|endoftext|>"],  # 有一个不想要的字符串
            eod_words=["<|endoftext|>"],  # 有一个停止符
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")


class StopWordsLogitsProcessor(LogitsProcessor):
    """
    用来停止生成的, 指定了一些停止符, 只支持停止序列的列表
    :class:`transformers.LogitsProcessor` that enforces that when specified sequences appear, stop geration.

    Args:
        stop_words_ids (:obj:`List[List[int]]`):
            List of list of token ids of stop ids. In order to get the tokens of the words
            that should not appear in the generated text, use :obj:`tokenizer(bad_word,
            add_prefix_space=True).input_ids`.
        eos_token_id (:obj:`int`):
            The id of the `end-of-sequence` token.
    """

    def __init__(self, stop_words_ids: Iterable[Iterable[int]], eos_token_id: int):
        """_summary_

        Args:
            stop_words_ids (Iterable[Iterable[int]]): 是停止序列的列表, 是个二维列表, 不是停止字符串的列表, 是 token_seq 的列表
            eos_token_id (int): 用来停止生成的 token id

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """

        if not isinstance(stop_words_ids, List) or len(stop_words_ids) == 0:
            raise ValueError(
                f"`stop_words_ids` has to be a non-emtpy list, but is {stop_words_ids}."
            )
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in stop_words_ids):
            raise ValueError(
                f"`stop_words_ids` has to be a list of lists, but is {stop_words_ids}."
            )
        if any(
            any(
                (not isinstance(token_id, (int, np.integer)) or token_id < 0)
                for token_id in stop_word_ids
            )
            for stop_word_ids in stop_words_ids
        ):
            raise ValueError(
                f"Each list in `stop_words_ids` has to be a list of positive integers, but is {stop_words_ids}."
            )

        # 过滤掉 [eos_token_id]
        self.stop_words_ids = list(
            filter(
                lambda bad_token_seq: bad_token_seq != [eos_token_id], stop_words_ids
            )
        )
        # eos_token_id 用来停止生成
        self.eos_token_id = eos_token_id
        # 停止字符串的长度应该大于 0
        for stop_token_seq in self.stop_words_ids:
            assert (
                len(stop_token_seq) > 0
            ), "Stop words token sequences {} cannot have an empty list".format(
                stop_words_ids
            )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """调用

        Args:
            input_ids (torch.LongTensor): 应该是所有的历史 id, 是批次的, 所以形状是 (batch_size, length)
            scores (torch.FloatTensor): 当前的分数, 用来产生下一个 token

        Returns:
            torch.FloatTensor: _description_
        """
        stopped_samples = self._calc_stopped_samples(input_ids)
        for i, should_stop in enumerate(stopped_samples):
            if should_stop:
                # 当符合条件时, 将 eos_token_id 的概率设置为最大
                scores[i, self.eos_token_id] = float(2**15)
        return scores

    def _tokens_match(self, prev_tokens: torch.LongTensor, tokens: List[int]) -> bool:
        """是否匹配

        Args:
            prev_tokens (torch.LongTensor): 是个 token 列表
            tokens (List[int]): _description_

        Returns:
            bool: _description_
        """
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        elif len(tokens) > len(prev_tokens):
            # 长度不符合, 自然是不匹配
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False
        elif prev_tokens[-len(tokens):].tolist() == tokens:
            # 符合条件, 最后 N 个 token 匹配上了
            # if tokens match
            return True
        else:
            return False

    def _calc_stopped_samples(self, prev_input_ids: Iterable[int]) -> Iterable[int]:
        """计算是否匹配

        Args:
            prev_input_ids (Iterable[int]): 形状是 (batch_size, length)

        Returns:
            Iterable[int]: _description_
        """
        stopped_samples = []
        # prev_input_ids_slice 是当前的历史序列
        for prev_input_ids_slice in prev_input_ids:
            match = False
            # 遍历所有的停止序列
            for stop_token_seq in self.stop_words_ids:
                # 是否能匹配上
                if self._tokens_match(prev_input_ids_slice, stop_token_seq):
                    # if tokens do not match continue
                    match = True
                    break
            stopped_samples.append(match)

        return stopped_samples


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """This function has been mostly taken from huggingface conversational
    ai code at
        https://medium.com/huggingface/how-to-build-a-state-of-the-art-
             conversational-ai-with-transfer-learning-2d818ac26313"""

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2
