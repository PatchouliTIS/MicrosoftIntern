
import numpy as np
from transformers import AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast
from typing import Dict, Any
import orjson
import re

import unicodedata
from urllib.parse import unquote

from vllm import SamplingParams


def load_tokenizer(
    model_name_or_path, tokenizer_type="auto", fast_tokenizer=False, padding_side="left"
):
    assert tokenizer_type in (
        "llama",
        "auto",
    ), f"Unsupported tokenizer type: {tokenizer_type}"

    print(
        f"Loading tokenizer ({tokenizer_type}) from {model_name_or_path}. Use fast tokenizer: {fast_tokenizer}."
    )
    if tokenizer_type == "llama" and fast_tokenizer is False:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, use_fast=False, padding_side=padding_side
        )
    elif tokenizer_type == "llama" and fast_tokenizer is True:
        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_name_or_path, use_fast=True, padding_side=padding_side
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=fast_tokenizer, padding_side=padding_side
        )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer ({tokenizer.__class__.__name__}) loaded.")
    return tokenizer


_pattern = re.compile(r"<[^>]+>", re.S)


def remove_control_char(input_string):
    return "".join(
        [c for c in input_string if not unicodedata.category(c).startswith("C")]
    )


def clean_sentence(line, pattern=None, is_url: bool = False):
    if pattern is None:
        pattern = _pattern

    if is_url:
        line = unquote(line)  # decode url
        return line
    else:
        line = pattern.sub(
            "",
            line
            # remove_control_char(line)  # remove control characters
            .replace("\ue000", "")  # highlighted tags
            .replace("\ue001", "")
            .replace("<strong>", "")  # highlighted tags
            .replace("</strong>", "")
            .replace("<span>", "")
            .replace("</span>", "")
            .replace("#HASH#N#HASH#", " ")
            .replace("#HASHHASH#", " ")
            .replace("#HASH#", " ")
            .replace("#N#", "\n")
            .replace("#TAB#", " "),
        )
        # line = re.sub(' +',' ',line)
        return line


def clean_prompt(prompt):
    prompt = prompt.replace("�", " ")
    prompt = clean_sentence(prompt, is_url=False)
    return prompt


def truncate_prompt(
        prompt,
        tokenizer,
        max_seq_len=4096,
        buffer=1000,
    ):
    print(f"Truncate prompt with max_seq_len:{max_seq_len} and buffer:{buffer}")
    tail_mark = '#Query Guidelines Formatting\n'
    start_idx = prompt.find(tail_mark)
    if start_idx == -1:
        return prompt
    prompt = clean_prompt(prompt)
    body = prompt[:start_idx].rstrip()
    tail_context = prompt[start_idx:].rstrip()
    tail_token_len = len(tokenizer.encode(tail_context, add_special_tokens=False))
    current_len = len(tokenizer.tokenize(prompt))
    if current_len <= max_seq_len - tail_token_len - buffer:
        print(f"Untouched prompt length:{current_len}")
        return prompt
    else:
        # print("Truncate prompt")
        body_tokens = tokenizer.encode(body, add_special_tokens=False)
        available_tokens_for_body = max_seq_len - tail_token_len - buffer
        body_tokens = body_tokens[:available_tokens_for_body]
        body = tokenizer.decode(body_tokens)
        body = body.replace("�", "") + "\n"
        
        final_prompt = (
            body + tail_context
        )
        print(f"Truncated prompt to {len(body_tokens) + tail_token_len} tokens.")
        return final_prompt


class FastTokenizer(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = load_tokenizer(
            model_name_or_path=self.args.model_path,
            tokenizer_type="auto",
            fast_tokenizer=True,
            padding_side="left",
        )

    def tokenize(self, prompts) -> Dict[str, Any]:
        
        input_ids = self.tokenizer([truncate_prompt(prompt, self.tokenizer, self.args.max_model_lens, self.args.max_new_tokens) for prompt in prompts], add_special_tokens=True)["input_ids"]
        # input_ids_np = [np.array(line, dtype="int32") for line in input_ids]  max_model_lens self.args.max_model_lens  self.args.max_new_tokens
        # print(f"input_ids:{input_ids}\n class:{type(input_ids)}\n--------------------------\n")
        sampling_config = SamplingParams(
            temperature=0,
            top_p=1.0,
            max_tokens=self.args.max_new_tokens,
            # stop_words_list=torch.from_numpy(stop_ids).cuda(),  # ?
        )

        return {
            "input_ids": input_ids,
            "sampling_config": sampling_config,
        }
