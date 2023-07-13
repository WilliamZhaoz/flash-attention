import math
import torch
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all
from flash_attn.bert_padding import unpad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_func


def attention_ref(q, k, v, attn_mask, dropout_p, upcast=False, causal=False):
    """
    Arguments:
        q, k, v: (batch_size, seqlen, nheads, head_dim)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """

    # seqlen_q = q.shape[1]
    # seqlen_k = k.shape[1]
    # seqlen_v = v.shape[1]
    d = q.shape[-1]
    scores = torch.einsum('bthd,bshd->bhts', q, k / math.sqrt(d))
    scores.masked_fill_(rearrange(~attn_mask, 'b s -> b 1 1 s'), float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop, v)
    # return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)
    return output.to(dtype=q.dtype)


torch.manual_seed(0)
repeats = 30
batch_size = 64
nheads = 16

seqlen_q = 1024
seqlen_k = 1024
seqlen_v = 1024

n = 1024
d = n // nheads
dropout_p = 0.1
causal = False
dtype = torch.float16
device = 'cuda'

q = torch.randn(batch_size, seqlen_q, n, device='cuda', dtype=dtype, requires_grad=True)
k = torch.randn(batch_size, seqlen_k, n, device='cuda', dtype=dtype, requires_grad=True)
v = torch.randn(batch_size, seqlen_v, n, device='cuda', dtype=dtype, requires_grad=True)
Wqkv = torch.nn.Linear(nheads * d, nheads * d, device=device, dtype=dtype)

lengths_q = torch.randint(seqlen_q - 20, seqlen_q, (batch_size, 1), device='cuda')
attention_mask_bool_q = repeat(torch.arange(seqlen_q, device='cuda'), 's -> b s', b=batch_size) < lengths_q
attention_mask_q = torch.zeros(batch_size, seqlen_q, device='cuda', dtype=dtype)
attention_mask_q[~attention_mask_bool_q] = -10000.0
attention_mask_q = rearrange(attention_mask_q, 'b s -> b 1 1 s')

lengths_k = torch.randint(seqlen_k - 20, seqlen_k, (batch_size, 1), device='cuda')
attention_mask_bool_k = repeat(torch.arange(seqlen_k, device='cuda'), 's -> b s', b=batch_size) < lengths_k
attention_mask_k = torch.zeros(batch_size, seqlen_k, device='cuda', dtype=dtype)
attention_mask_k[~attention_mask_bool_k] = -10000.0
attention_mask_k = rearrange(attention_mask_k, 'b s -> b 1 1 s')

lengths_v = torch.randint(seqlen_v - 20, seqlen_v, (batch_size, 1), device='cuda')
attention_mask_bool_v = repeat(torch.arange(seqlen_v, device='cuda'), 's -> b s', b=batch_size) < lengths_v
attention_mask_v = torch.zeros(batch_size, seqlen_v, device='cuda', dtype=dtype)
attention_mask_v[~attention_mask_bool_v] = -10000.0
attention_mask_v = rearrange(attention_mask_v, 'b s -> b 1 1 s')

q_unpad, indices, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(q, attention_mask_bool_q)
k_unpad, indices, cu_seqlens_k, max_seqlen_in_batch_k = unpad_input(k, attention_mask_bool_k)
v_unpad, indices, cu_seqlens_v, max_seqlen_in_batch_v = unpad_input(v, attention_mask_bool_v)

q_unpad = rearrange(Wqkv(q_unpad), 'nnz (h d) -> nnz h d', h=nheads).detach().requires_grad_()
k_unpad = rearrange(Wqkv(k_unpad), 'nnz (h d) -> nnz h d', h=nheads).detach().requires_grad_()
v_unpad = rearrange(Wqkv(v_unpad), 'nnz (h d) -> nnz h d', h=nheads).detach().requires_grad_()

q = rearrange(Wqkv(q), 'b s (h d) -> b s h d', h=nheads).detach().requires_grad_()
k = rearrange(Wqkv(k), 'b s (h d) -> b s h d', h=nheads).detach().requires_grad_()
v = rearrange(Wqkv(v), 'b s (h d) -> b s h d', h=nheads).detach().requires_grad_()

fn = lambda q_unpad, k_unpad, v_unpad: flash_attn_unpadded_func(q_unpad,
                                                                k_unpad,
                                                                v_unpad,
                                                                cu_seqlens_q,
                                                                cu_seqlens_k,
                                                                max_seqlen_in_batch_q,
                                                                max_seqlen_in_batch_k,
                                                                dropout_p,
                                                                causal=causal)
benchmark_all(fn, q_unpad, k_unpad, v_unpad, repeats=repeats, desc='FlashAttention')
fn = lambda q, k, v: attention_ref(q, k, v, attention_mask_bool_q, dropout_p, causal=causal)
benchmark_all(fn, q, k, v, repeats=repeats, desc='PyTorch Standard Attention')
