import torch
import triton
import triton.language as tl


@triton.jit
def _apply_rope_kernel(
    T, FREQS, OUT, 
    DIM, BH, FREQS_DIM,
    BLOCK_SIZE: tl.constexpr
):
    # Compute linear index for this thread
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK_SIZE)

    # Compute original 2D indices (seq_idx and dim_idx) from linear index
    seq_idx = pid // BH

    t = tl.load(T + pid * BLOCK_SIZE + idx)

    freqs = tl.load(FREQS + seq_idx * FREQS_DIM + idx)
    freqs = tl.where(idx < FREQS_DIM, freqs, 0.0)

    # Apply RoPE
    cos_ = tl.cos(freqs)
    sin_ = tl.sin(freqs)
    t_cos = t * cos_

    # Rotate half logic
    half_dim = FREQS_DIM // 2
    new_idx = tl.where(idx >= half_dim, idx - half_dim, idx + half_dim)
    half_t = tl.load(T + pid * BLOCK_SIZE + new_idx)
    rotated_t = tl.where(new_idx >= half_dim, -half_t, half_t)

    res = tl.where(idx < FREQS_DIM, t_cos + rotated_t * sin_, t)
    # Write back to output
    tl.store(OUT + pid * BLOCK_SIZE + idx, res)

def apply_rotary_pos_emb_triton(
    t: torch.Tensor, 
    freqs: torch.Tensor, 
    BLOCK_SIZE=128
) -> torch.Tensor:
    # Reshape and pad input tensor and freqs
    seq_len, b, h, d = t.shape
    t = t.view(seq_len, -1).contiguous()
    freqs = freqs.view(seq_len, -1).contiguous()

    # Allocate output tensor
    out = torch.empty_like(t)

    # Calculate 1D grid size
    BLOCK_SIZE = d
    total_tasks = seq_len * b * h * d
    grid = ((total_tasks + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch kernel with 1D grid
    _apply_rope_kernel[grid](
        t, freqs, out, 
        seq_len, b * h, freqs.shape[-1],
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out.view(seq_len, b, h, d)
