"""PyTorch Native Fusion Attention Implementation."""

import torch
import torch.nn.functional as F

from ..function import MojoFunction

class MojoFusionAttentionFunction(MojoFunction):
    """PyTorch native Fusion Attention using scaled_dot_product_attention."""

    @staticmethod
    def forward(
        ctx,
        query, key, value,
        actual_seq_qlen, actual_seq_kvlen,
        head_num,
        scale=1.0,
        dropout_p=0.0,
        atten_mask=None,
        is_varlen=True,
        is_causal=True,
        **kwargs,
    ):
        if is_varlen:
            attn_out = _varlen_attention(
                query, key, value, actual_seq_qlen, actual_seq_kvlen,
                scale, dropout_p, atten_mask, is_causal,
            )
        else:
            attn_out = _sdpa(query, key, value, scale, dropout_p, atten_mask, is_causal)

        ctx.save_for_backward(query, key, value, atten_mask)
        ctx.is_varlen = is_varlen
        ctx.scale = scale
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_kvlen = actual_seq_kvlen
        ctx.is_causal = is_causal
        return attn_out

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, atten_mask = ctx.saved_tensors

        with torch.enable_grad():
            q, k, v = [t.detach().requires_grad_(True) for t in (query, key, value)]
            if ctx.is_varlen:
                out = _varlen_attention(
                    q, k, v, ctx.actual_seq_qlen, ctx.actual_seq_kvlen,
                    ctx.scale, 0.0, atten_mask, ctx.is_causal,
                )
            else:
                out = _sdpa(q, k, v, ctx.scale, 0.0, atten_mask, ctx.is_causal)
            out.backward(grad_output)

        return q.grad, k.grad, v.grad, None, None, None, None, None, None, None, None


def _sdpa(query, key, value, scale, dropout_p, atten_mask, is_causal):
    """Scaled dot-product attention wrapper."""
    q_head_num, kv_head_num = query.shape[1], key.shape[1]
    enable_gqa = (q_head_num != kv_head_num)
    
    if is_causal and atten_mask is None:
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=True, scale=scale, enable_gqa=enable_gqa
        )
    # Convert bool mask: NPU uses True=masked, SDPA uses -inf=masked
    if atten_mask is not None and atten_mask.dtype == torch.bool:
        atten_mask = atten_mask.float().masked_fill(atten_mask, float('-inf'))
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=atten_mask, dropout_p=dropout_p, scale=scale, enable_gqa=enable_gqa
    )


def _varlen_attention(query, key, value, actual_seq_qlen, actual_seq_kvlen,
                    scale, dropout_p, atten_mask, is_causal):
    """Variable-length attention for TND layout."""
    # Convert cumulative lengths to per-sequence lengths
    seq_qlen = actual_seq_qlen if isinstance(actual_seq_qlen, list) else actual_seq_qlen.tolist()
    seq_kvlen = actual_seq_kvlen if isinstance(actual_seq_kvlen, list) else actual_seq_kvlen.tolist()
    q_lens = [seq_qlen[0]] + [seq_qlen[i] - seq_qlen[i-1] for i in range(1, len(seq_qlen))]
    kv_lens = [seq_kvlen[0]] + [seq_kvlen[i] - seq_kvlen[i-1] for i in range(1, len(seq_kvlen))]

    outputs, q_off, kv_off = [], 0, 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        # Extract and reshape: TND -> BNSD (1, H, S, D)
        q = query[q_off:q_off + q_len].transpose(0, 1).unsqueeze(0)
        k = key[kv_off:kv_off + kv_len].transpose(0, 1).unsqueeze(0)
        v = value[kv_off:kv_off + kv_len].transpose(0, 1).unsqueeze(0)

        # Slice mask if provided
        mask = atten_mask[:q_len, :kv_len] if atten_mask is not None else None
        out = _sdpa(q, k, v, scale, dropout_p, mask, is_causal and mask is None)

        # Reshape back: BNSD -> TND
        outputs.append(out.squeeze(0).transpose(0, 1))
        q_off += q_len
        kv_off += kv_len

    return torch.cat(outputs, dim=0)