# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
import numpy as np
import torch
# import time
import triton
import triton.language as tl
import string
# import os
# os.environ['TRITON_INTERPRET'] = '1'

# from flash_attn import flash_attn_varlen_func
# import pycuda.autoprimaryctx
# from pycuda.compiler import SourceModule


@triton.jit
def _triton_matmul_kernel(
    A, B, C,
    stride_az, stride_ah, stride_am, stride_ak,
    stride_bz, stride_bh, stride_bk, stride_bn,
    stride_cz, stride_ch, stride_cm, stride_cn,
    Z,H,
    M, K, N,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    dtype: tl.constexpr,
) :
    # print(input_dtype)
    # -- grid id --
    pid = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    
    # GROUP check
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # offset pointers for (batch, head)
    a_init = A + off_z * stride_az + off_h * stride_ah
    b_init = B + off_z * stride_bz + off_h * stride_bh
    c_init = C + off_z * stride_cz + off_h * stride_ch

    offs_a_base = tl.arange(0, BLOCK_M)
    offs_am = (pid_m * BLOCK_M + offs_a_base) % M
    offs_b_base = tl.arange(0, BLOCK_N)
    offs_bn = (pid_n * BLOCK_N + offs_b_base) % N
    offs_k = tl.arange(0, BLOCK_DMODEL)
    
    # load a, b
    a_ptrs = a_init + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_init + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # loop for K dimension
    for k in range(0, K, BLOCK_DMODEL):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other = 0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other = 0.0)
        
        acc = tl.dot(a, b, acc)
        
        a_ptrs += BLOCK_DMODEL * stride_ak
        b_ptrs += BLOCK_DMODEL * stride_bk
        
    c = acc.to(dtype)
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_init + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _triton_attn(
    Q, K, V, sm_scale,
    dropout_p,
    seed,
    offset,
    # L, 
    O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr, IS_DROPOUT: tl.constexpr, LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # print(input_dtype)
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    O += off_z * stride_oz + off_h * stride_oh
    # L += (off_z * H + off_h) * M # l's shape is (B, H, M)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    if IS_DROPOUT:
        rowblock_base = off_z * H * M * N + off_h * M * N + start_m * BLOCK_M * N
        offs_rng_base = offset + rowblock_base
        offs_rng_base += tl.arange(0, BLOCK_M)[:, None] * N
        offs_rng_base += tl.arange(0, BLOCK_N)[None, :]

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
    # l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    # tl.float32
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")

    #Dot I trick: to place q in registers, it saves shared memory
    if BLOCK_DMODEL < 128:
        I = tl.where(offs_k[:, None] == offs_k,
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
                     tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
        q = tl.dot(q, I).to(input_dtype)
    # else:
    #     I = tl.where(offs_m_base[:, None] == offs_m_base,
    #                  tl.full((BLOCK_M, BLOCK_M), 1.0, dtype=input_dtype),
    #                  tl.full((BLOCK_M, BLOCK_M), 0.0, dtype=input_dtype))
    #     q = tl.dot(I, q).to(input_dtype)

    # NOTE: Loop-Bound-For-N
    # The indices in m-dimension that this block may access is in `[start_m * BLOCK_M, (start_m + 1) * BLOCK_M)`.
    # According to the rule of causal masking, then max index in n-dimension that this block may access
    # is `P_SEQ + (start_m + 1) * BLOCK_M`.
    # However, the upper bound of index in n-dimension should never exceed the sequence length of k/v(`P_SEQ + N_CTX`).
    # `P_SEQ + (start_m + 1) * BLOCK_M` may be larger than `N`.
    # At this case, there would be illegal memory access when loading k & v tiles
    # if mask_n is not applied for loading(only when `DIVISIBLE_N`` is true).
    # See also https://github.com/FlagOpen/FlagAttention/pull/8
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    # loop over k, v and update accumulators
    offs_n_init = offs_n_base
    k_ptrs = K + (offs_k[:, None] * stride_vk + offs_n_init[None, :] * stride_vn) # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk) # (BLOCK_N, BLOCK_DMODEL)
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")

        # -- compute qk ---
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=input_dtype)
        s += tl.dot(q, k)

        # if not DIVISIBLE_N:
        #     s = tl.where(mask_n[None, :], s, float("-inf"))
        # if IS_CAUSAL:
        causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
        s = tl.where(causal_mask, s, float("-inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new) * qk_scale)
        p = tl.math.exp2(s * qk_scale - m_i_new[:, None] * qk_scale)

        # -- compute partial sumexpn before applying dropout
        p_sum = tl.sum(p, 1)

        # -- apply dropout --
        if IS_DROPOUT:
            offs_rng = start_n + offs_rng_base
            pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
            # tl.float32
            p *= pmask.to(tl.float32)

        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        acc += tl.dot(p.to(input_dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # write back l & o
    if IS_CAUSAL and LARGER_M:
        is_empty_line = (offs_m + P_SEQ) < 0
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        # l = tl.where(is_empty_line, float("-inf"), m_i * sm_scale + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        # l = m_i * sm_scale + tl.log(l_i) # log(normalizer)

    # -- scale o due to dropout
    if IS_DROPOUT:
        scale = 1.0 / (1.0 - dropout_p)
        acc *= scale

    if DIVISIBLE_M:
        # tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        # tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")


# @triton.autotune(
#    configs=[
#        triton.Config({}, num_stages=1, num_warps=4),
#        triton.Config({}, num_stages=1, num_warps=8),
#        triton.Config({}, num_stages=2, num_warps=4),
#        triton.Config({}, num_stages=2, num_warps=8),
#        triton.Config({}, num_stages=3, num_warps=4),
#        triton.Config({}, num_stages=3, num_warps=8),
#        triton.Config({}, num_stages=4, num_warps=4),
#        triton.Config({}, num_stages=4, num_warps=8),
#        triton.Config({}, num_stages=5, num_warps=4),
#        triton.Config({}, num_stages=5, num_warps=8),
#    ],
#    key=['N_CTX'],
# )

@triton.jit
def _triton_block_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, MAX_BLOCKS_PRE_ROW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PRE_ROW

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # TODO
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) # TODO
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32) # TODO
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen
    block_count = tl.minimum((start_m + 1) * BLOCK_M // BLOCK_N, MAX_BLOCKS_PRE_ROW)

    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        start_n = real_block_idx * BLOCK_N
        cols = start_n + offs_n
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # TODO
        # if start_n + BLOCK_N < seqlen:
        #     qk = tl.where(m_mask, qk, float("-inf"))
        # else:
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = alpha  # workaround some compiler bug # DEBUG l_i * 0 +
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)
    
   
    
@triton.jit
def _triton_pyramidkv_attn_fwd_kernel(
    Q, K, V, seqlens_q, seqlens_k, sm_scale,
    block_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, MAX_BLOCKS_PRE_ROW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen_q = tl.load(seqlens_q + off_hz // H)
    seqlen_k = tl.load(seqlens_k + off_hz // H)
    if start_m * BLOCK_M >= seqlen_q:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PRE_ROW

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # Create masks for valid Q and K entries
    m_mask = offs_m[:, None] < seqlen_q
    block_count = tl.minimum((start_m + 1) * BLOCK_M // BLOCK_N, MAX_BLOCKS_PRE_ROW)

    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        start_n = real_block_idx * BLOCK_N
        cols = start_n + offs_n
        
        # Create dynamic mask
        k_mask = cols < seqlen_k
        causal_mask = cols[None, :] <= offs_m[:, None]
        
        # Load k and v
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        
        # Compute attention scores
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        
        
        # Apply masks
        qk_mask = m_mask & causal_mask
        qk = tl.where(qk_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        
        # Compute scaling constant
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        
        # Scale and update accumulator
        acc_scale = l_i * 0 + alpha  # workaround for compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        
        # Update m_i and l_i
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Write back output
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def triton_matmul(
    q,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k,                 # [BATCH, N_HEADS, D_HEAD, N_CTX]
    block_size_M=128,
    block_size_N=64,
) -> torch.Tensor:
    # transposed k
    Lq, Lk= q.shape[-1], k.shape[-2]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros([q.shape[0], q.shape[1], q.shape[2], k.shape[-1]], dtype=q.dtype, device=q.device)
    grid = (triton.cdiv(q.shape[2], block_size_M) * triton.cdiv(k.shape[-1], block_size_N), q.shape[1], q.shape[0])
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    # dtype = tl.bfloat16
    _triton_matmul_kernel[grid](
        q, k, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1],
        q.shape[2], q.shape[3], k.shape[3],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=32,
        GROUP_SIZE_M=8,
        dtype=dtype,
        num_warps=4, num_stages=4,
        num_ctas=1
        )

    return o

def _triton_block_sparse_attention(
    q,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens,           # [BATCH, ]
    block_index,       # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), MAX_BLOCKS_PRE_ROW]
    sm_scale,
    block_size_M=64,
    block_size_N=64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    # dtype = tl.bfloat16
    _triton_block_sparse_attn_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        block_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        block_index.shape[-2], block_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )

    return o


def maybe_contiguous(x):
    # only when the inner most dimension is contiguous can LDGSTS be used
    # so inner-dimension contiguity is enforced.
    return x.contiguous() if x.stride(-1) != 1 else x

def _triton_pyramidkv_attention(
    q,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqqlens,          # [BATCH, ]
    seqklens,          # [BATCH, ]
    block_index,       # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), MAX_BLOCKS_PRE_ROW]
    sm_scale,
    block_size_M=64,
    block_size_N=64,
    mask=None,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    bsz, num_heads, q_len, head_dim = q.shape
    k_len = k.shape[-2]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    # o = torch.zeros([bsz, num_heads, q_len, k_len], dtype=q.dtype, device=q.device)
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float32
    # _triton_pyramidkv_attn_fwd_kernel[grid](
    #     q, k, v, seqqlens, seqklens, sm_scale,
    #     block_index,
    #     o,
    #     q.stride(0), q.stride(1), q.stride(2), q.stride(3),
    #     k.stride(0), k.stride(1), k.stride(2), k.stride(3),
    #     v.stride(0), v.stride(1), v.stride(2), v.stride(3),
    #     o.stride(0), o.stride(1), o.stride(2), o.stride(3),
    #     q.shape[0], q.shape[1], q.shape[2],
    #     block_index.shape[-2], block_index.shape[-1],
    #     BLOCK_M=block_size_M, BLOCK_N=block_size_N,
    #     BLOCK_DMODEL=Lk,
    #     dtype=dtype,
    #     num_warps=4, num_stages=2,
    # )
    # _triton_block_sparse_attn_fwd_kernel[grid](
    #     q, k, v, seqqlens, sm_scale,
    #     block_index,
    #     o,
    #     q.stride(0), q.stride(1), q.stride(2), q.stride(3),
    #     k.stride(0), k.stride(1), k.stride(2), k.stride(3),
    #     v.stride(0), v.stride(1), v.stride(2), v.stride(3),
    #     o.stride(0), o.stride(1), o.stride(2), o.stride(3),
    #     q.shape[0], q.shape[1], q.shape[2],
    #     block_index.shape[-2], block_index.shape[-1],
    #     BLOCK_M=block_size_M, BLOCK_N=block_size_N,
    #     BLOCK_DMODEL=Lk,
    #     dtype=dtype,
    #     num_warps=4, num_stages=2,
    # )
    dropout_p = 0.0
    seed = 0
    offset = 0
    P_SEQ = k_len - q_len
    larger_m = q_len > k_len
    num_groups = num_heads // k.shape[1]
    causal = True
    is_dropout = dropout_p > 0
    divisible_m = q_len % block_size_M == 0  
    divisible_n = k_len % block_size_N == 0
    q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)
    cpt_dtype = dtype
    _triton_attn[grid](
                    q, k, v, sm_scale,
                    dropout_p, seed, offset,
                    o,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                    bsz, num_heads, q_len, k_len, P_SEQ, num_groups, 
                    BLOCK_M=block_size_M, BLOCK_N=block_size_N, BLOCK_DMODEL=Lk,
                    IS_CAUSAL=causal, IS_DROPOUT=is_dropout, LARGER_M=larger_m,
                    DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                    num_warps=4, num_stages=2,
                )

    return o

def _build_block_index(
    query: torch.Tensor,     # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,       # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 128,
    block_size_N: int = 128,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    query_pool = query.reshape((batch_size, num_heads, -1, block_size_M, head_dim)).mean(dim=-2)
    key_pool = key.reshape((batch_size, num_heads, -1, block_size_N, head_dim)).mean(dim=-2)
    arange_M = torch.arange(query_pool.shape[-2], dtype=torch.int32, device=query.device) * block_size_M
    arange_N = torch.arange(key_pool.shape[-2], dtype=torch.int32, device=key.device) * block_size_N
    p_pool = torch.einsum(f'bhmk, bhnk -> bhmn', query_pool, key_pool)
    p_pool = p_pool.where(arange_M[None, None, :, None] >= arange_N[None, None, None, :], -torch.inf)
    # print(f"top_k: {top_k}\tactual size:{context_size // block_size_N}")
    top_k = min(top_k, context_size // block_size_N)
    block_index = torch.topk(p_pool, top_k, dim=-1).indices.to(torch.int32).sort(dim=-1).values
    return block_index


def block_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
    block_index: torch.Tensor = None,
    seqlens: torch.Tensor = None,
):
    # start_time = time.perf_counter()
    top_k = 4
    batch_size, num_heads, context_size, head_dim = query.shape
    # pad = block_size_M - (query.shape[2] & (block_size_M - 1))
    # query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    # key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    # value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])
    # seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    sm_scale = head_dim ** -0.5
    context_size = seqlens[0]
    # block_index = _build_block_index(query, key, top_k, block_size_N, block_size_N)
    out = _triton_block_sparse_attention(query, key, value, seqlens, block_index, sm_scale, block_size_M, block_size_N)
    # elapsed_time = (time.perf_counter() - start_time) * 1000
    # print(f"Block sparse attention time: {elapsed_time} ms")
    return out[..., :context_size, :]



import math
import torch.nn as nn
import torch.nn.functional as F
def pyramidkv_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    layer_idx: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
    window_size: int = 8,
    max_capacity_prompt: int = 256 + 64,
    kernel_size: int = 7,
    pooling: string = 'maxpool',
):
    num_hidden_layers = 32
    beta = 20
    
    # check if prefix phase
    assert query.shape[-2] == key.shape[-2]
    bsz, num_heads, q_len, head_dim = query.shape
    # import triton
    # max_capacity_prompt = triton.next_power_of_2((int)(q_len / 2))
    # max_capacity_prompt = q_len
    # # TODO calculate max_capacity_prompt
    # min_num = (max_capacity_prompt - window_size) // beta
    # max_num = (max_capacity_prompt - window_size) * 2 - min_num
    
        
    # if max_num >= q_len - window_size:
    #     max_num = q_len - window_size
    #     min_num = (max_capacity_prompt - window_size) * 2 - max_num
    # global count
    # steps = (max_num - min_num) // num_hidden_layers
    # max_capacity_prompt = max_num - layer_idx * steps
    # # print(f"PyramidKV max_capacity_prompt {max_capacity_prompt}\tglobal count: {global_count}\tq_len:{q_len}\tlayer_idx:{layer_idx}")
    # # print(f"Initially Matrix --> key shape:{key.shape}\tvalue shape:{value.shape}")
    # # TODO: topk selection
    # # max_capacity_prompt = q_len
    
    # if q_len < max_capacity_prompt:
    #     # print(f"untouched  --> key shape:{key_states.shape}\tvalue shape:{value_states.shape}")
    #     key_states = key
    #     value_states = value
    # elif q_len < (max_capacity_prompt - window_size) * 2 :
    #     attn_weights = torch.matmul(query[..., -window_size:, :], key.transpose(2, 3)) / math.sqrt(head_dim)
    #     # attn_weights = torch.einsum('...ij,...jk->...ik', query[..., -window_size:, :], key.transpose(2, 3)) / math.sqrt(head_dim)
    #     mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    #     mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    #     mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0).to(attn_weights.device)
    #     attention_mask = mask[None, None, :, :]
    #     attn_weights[:, :, -window_size:, -window_size:] += attention_mask
    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    #     attn_weights_sum = attn_weights[:, :, -window_size:, : -window_size].sum(dim = -2)
    #     if pooling == 'avgpool':
    #         attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
    #     elif pooling == 'maxpool':
    #         attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
    #     else:
    #         raise ValueError('Pooling method not supported')
    #     # attn_cache = attn_weights_sum
    #     indices = attn_cache.topk(max_capacity_prompt - window_size, dim=-1).indices
    #     indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    #     k_past_compress = key[:, :, :-window_size, :].gather(dim = 2, index = indices)
    #     v_past_compress = value[:, :, :-window_size, :].gather(dim = 2, index = indices)
    #     k_cur = key[:, :, -window_size:, :]
    #     v_cur = value[:, :, -window_size:, :]
    #     key_states = torch.cat([k_past_compress, k_cur], dim = 2)
    #     value_states = torch.cat([v_past_compress, v_cur], dim = 2)
    #     # print(f"after cat 1 --> key shape:{key_states.shape}\tvalue shape:{value_states.shape}")
    #     # return key_states, value_states
    # else:
    #     attn_weights = torch.matmul(query[..., -window_size:, :], key.transpose(2, 3)) / math.sqrt(head_dim)
    #     mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    #     mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    #     mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0).to(attn_weights.device)
    #     attention_mask = mask[None, None, :, :]
    #     attn_weights[:, :, -window_size:, -window_size:] += attention_mask
    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    #     attn_weights_sum = attn_weights[:, :, -window_size:, : -window_size].sum(dim = -2)
    #     if pooling == 'avgpool':
    #         attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
    #     elif pooling == 'maxpool':
    #         attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
    #     else:
    #         raise ValueError('Pooling method not supported')
    #     # attn_cache = attn_weights_sum
    #     indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
    #     # indices = attn_cache.topk(max_capacity_prompt - window_size, dim=-1).indices
    #     indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    #     k_past_compress = key[:, :, :-window_size, :].gather(dim = 2, index = indices)
    #     v_past_compress = value[:, :, :-window_size, :].gather(dim = 2, index = indices)
    #     k_cur = key[:, :, -window_size:, :]
    #     v_cur = value[:, :, -window_size:, :]
    #     key_states = torch.cat([k_past_compress, k_cur], dim = 2)
    #     value_states = torch.cat([v_past_compress, v_cur], dim = 2)
    #     # print(f"after cat 2 -->key shape:{key_states.shape}\tvalue shape:{value_states.shape}")
    
    # key = key_states
    # value = value_states
    
    k_len = key.shape[-2]
    
    # print(f"key_status:{key_states.shape}\tvalue_status:{value_states.shape}\tquery_status:{query.shape}")
    
    # Padding
    # pad = block_size_M - (query.shape[2] & (block_size_M - 1))
    # key_pad = block_size_M - (key_states.shape[2] & (block_size_N - 1))
    # # print(f"key_status:{key_states.shape}\tpad:{key_pad}\tquery_status:{query.shape}\tpad:{pad}")
    # query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    # key = torch.nn.functional.pad(key_states, [0, 0, 0, key_pad, 0, 0, 0, 0])
    # value = torch.nn.functional.pad(value_states, [0, 0, 0, key_pad, 0, 0, 0, 0])
    seqlens = torch.tensor([q_len], dtype=torch.int32, device=query.device)
    seqklens = torch.tensor([k_len], dtype=torch.int32, device=query.device)
    sm_scale = head_dim ** -0.5
    # calculate
    block_index = _build_dog_block_index(query, key, block_size_M, block_size_N)
    out = _triton_pyramidkv_attention(query, key, value, seqlens, seqklens, block_index, sm_scale, block_size_M, block_size_N)
    return out[..., :q_len, :]



def snapkv_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    layer_idx: int,
    block_size_M: int = 128,
    block_size_N: int = 32,
    window_size: int = 128,
    max_capacity_prompt: int = 256 + 64,
    kernel_size: int = 5,
    pooling: string = 'maxpool',
):
    bsz, num_heads, q_len, head_dim = query.shape
    # # # TODO: Implement snapkv_attention
    # beta = 20
    # num_hidden_layers = 32
    # # check if prefix phase
    # assert query.shape[-2] == key.shape[-2]
    # bsz, num_heads, q_len, head_dim = query.shape
    # # import triton
    # # max_capacity_prompt = triton.next_power_of_2((int)(q_len / 2))
    # max_capacity_prompt = (int)(q_len * 0.7)
    # window_size = (int)(max_capacity_prompt * 0.5)
    # # min_num = (max_capacity_prompt - window_size) // beta
    # # max_num = (max_capacity_prompt - window_size) * 2 - min_num
    
        
    # # if max_num >= q_len - window_size:
    # #     max_num = q_len - window_size
    # #     min_num = (max_capacity_prompt - window_size) * 2 - max_num
    # # global count
    # # steps = (max_num - min_num) // num_hidden_layers
    # # max_capacity_prompt = max_num - layer_idx * steps
    # # print(f"PyramidKV max_capacity_prompt {max_capacity_prompt}\tglobal count: {global_count}\tq_len:{q_len}\tlayer_idx:{layer_idx}")
    # # # print(f"Initially Matrix --> key shape:{key.shape}\tvalue shape:{value.shape}")
    # # # TODO: topk selection
    # if q_len < max_capacity_prompt:
    #     # print(f"untouched  --> key shape:{key_states.shape}\tvalue shape:{value_states.shape}")
    #     key_states = key
    #     value_states = value
    # else:
    #     key_states = key
    #     value_states = value
    #     attn_weights = torch.matmul(query[..., -window_size:, :], key.transpose(2, 3)) / math.sqrt(head_dim)
    #     # attn_weights = torch.einsum('...ij,...jk->...ik', query[..., -window_size:, :], key.transpose(2, 3)) / math.sqrt(head_dim)
    #     mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    #     mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
    #     mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    #     mask.to(attn_weights.device)
    #     attention_mask = mask[None, None, :, :]
    #     attn_weights[:, :, -window_size:, -window_size:] += attention_mask
    #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    #     attn_weights_sum = attn_weights[:, :, -window_size:, : -window_size].sum(dim = -2)
    #     if pooling == 'avgpool':
    #         attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
    #     elif pooling == 'maxpool':
    #         attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = kernel_size, padding=kernel_size//2, stride=1)
    #     else:
    #         raise ValueError('Pooling method not supported')
    #     # attn_cache = attn_weights_sum
    #     indices = attn_cache.topk(max_capacity_prompt - window_size, dim=-1).indices
    #     indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    #     k_past_compress = key[:, :, :-window_size, :].gather(dim = 2, index = indices)
    #     v_past_compress = value[:, :, :-window_size, :].gather(dim = 2, index = indices)
    #     k_cur = key[:, :, -window_size:, :]
    #     v_cur = value[:, :, -window_size:, :]
    #     key_states = torch.cat([k_past_compress, k_cur], dim = 2)
    #     value_states = torch.cat([v_past_compress, v_cur], dim = 2)
    
    
    # key = key_states
    # value = value_states
    
    k_len = key.shape[-2]
    # key_states = key
    # value_states = value
    # TODO: still have to pad， but in the end
    # pad = block_size_M - (query.shape[2] & (block_size_M - 1))
    # key_pad = block_size_M - (key_states.shape[2] & (block_size_M - 1))
    # query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    # key = torch.nn.functional.pad(key_states, [0, 0, 0, key_pad, 0, 0, 0, 0])
    # value = torch.nn.functional.pad(value_states, [0, 0, 0, key_pad, 0, 0, 0, 0])
    # print(f"after cat&pad -->key shape:{key_states.shape}\tvalue shape:{value_states.shape}\tquery shape:{query.shape}")
    seqlens = torch.tensor([q_len], dtype=torch.int32, device=query.device)
    seqklens = torch.tensor([k_len], dtype=torch.int32, device=query.device)
    # seqklens = torch.tensor([key_states.shape[2]], dtype=torch.int32, device=query.device)
    sm_scale = head_dim ** -0.5
    
    # calculate
    block_index = _build_dog_block_index(query, key, block_size_N, block_size_N)
    
    # TODO
    out = _triton_pyramidkv_attention(query, key, value, seqlens, seqklens, block_index, sm_scale, block_size_M, block_size_N)
    return out[..., :q_len, :]

def _build_dog_block_index(
    query: torch.Tensor,     # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,       # [BATCH, N_HEADS, N_CTX, D_HEAD]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = key.shape
    q_ctx_len = query.shape[2]
    # do not need to mean
    num_blocks_M = (q_ctx_len + block_size_M - 1) // block_size_M
    num_blocks_N = (context_size + block_size_N - 1) // block_size_N
    
    # Create a simple block index pattern (sequential or any predefined pattern)
    block_index = torch.arange(num_blocks_N, device=key.device).repeat(batch_size, num_heads, num_blocks_M, 1)
    # do not need to topk
    return block_index


