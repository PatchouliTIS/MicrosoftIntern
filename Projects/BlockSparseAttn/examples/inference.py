import os
import torch
from vllm import LLM, SamplingParams
from minference import MInference


class FastInference(object):
    def __init__(self, tokenizer, args):
        print(f"In FastInference ---> current torch.cuda.device_count:{torch.cuda.device_count()}")
        tokenizer_mode = "auto" if args.use_fast_tokenizer else "slow"
        self.llm = LLM(
            model=args.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            tokenizer_mode=tokenizer_mode,
            gpu_memory_utilization=0.9,
            dtype=args.model_dtype,
            
            max_model_len=args.max_model_lens,
            # disable_log_stats=True,
            
            # debug
            enforce_eager=True,
            
            # enbale prefix caching
            # enable_prefix_caching=True,
            # disable_sliding_window=True,
        )
        
        
        # Patch MInference Module
        minference_patch = MInference(attn_type="vllm", model_name=args.model_path, config_path="/workspace/MInference/minference/configs/Mistral_7B_BGM_kv_out_v32_fit_o_best_pattern.json")
        self.llm = minference_patch(self.llm)
        
        self.llm.set_tokenizer(tokenizer)
        print("use repetiion penalty")

        self.sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            ignore_eos=False,
            use_beam_search=False,
            max_tokens = args.max_new_tokens,
            # repetition_penalty=1.2,
        )
