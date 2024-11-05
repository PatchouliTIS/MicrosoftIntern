from dataclasses import dataclass

@dataclass
class ModelArgs:
    seed: int = 42
    model_type: str = "auto"
    model_path: str = "/workspace/rejmodel/rej16bit/mt_v632_16bit_rej_GPTQ_int4"   # "/workspace/model/oaas_models"   "/workspace/Mistral-7B-v0.2"
    use_fast_tokenizer: bool = True
    max_model_lens: int = 8000
    max_num_examples: int = None
    sample_method: str = "topk"
    max_new_tokens: int = 600
    eval_batch_size: int = 1
    use_cache: bool = True
    model_dtype: str = "float16"
    query_path: str = "/workspace/model/query/search_harder_intent.jsonl"
