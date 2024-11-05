# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL2PATH = {
    "gradientai/Llama-3-8B-Instruct-262k": os.path.join(
        BASE_DIR, "Llama_3_8B_Instruct_262k_kv_out_v32_fit_o_best_pattern.json"
    ),
    "gradientai/Llama-3-8B-Instruct-Gradient-1048k": os.path.join(
        BASE_DIR, "Llama_3_8B_Instruct_262k_kv_out_v32_fit_o_best_pattern.json"
    ),
    "gradientai/Llama-3-8B-Instruct-Gradient-4194k": os.path.join(
        BASE_DIR, "Llama_3_8B_Instruct_262k_kv_out_v32_fit_o_best_pattern.json"
    ),
    "meta-llama/Meta-Llama-3.1-8B-Instruct": os.path.join(
        BASE_DIR, "Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern.json"
    ),
    "gradientai/Llama-3-70B-Instruct-Gradient-262k": os.path.join(
        BASE_DIR, "Llama_3_70B_Instruct_262k_kv_out_v32_fit_o_best_pattern.json"
    ),
    "gradientai/Llama-3-70B-Instruct-Gradient-1048k": os.path.join(
        BASE_DIR, "Llama_3_70B_Instruct_262k_kv_out_v32_fit_o_best_pattern.json"
    ),
}


def get_support_models():
    return list(MODEL2PATH.keys())