import torch
import triton
import triton.language as tl


@triton.jit
def rope_fw(
    T, FREQS, OUT, 
    DIM, FREQS_DIM,
    BLOCK_SIZE: tl.constexpr
):
    # Compute index for this thread
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK_SIZE)
    seq_idx = pid // (DIM // BLOCK_SIZE)

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

    out = tl.where(idx < FREQS_DIM, t_cos + rotated_t * sin_, t)
    # Write back to output
    tl.store(OUT + pid * BLOCK_SIZE + idx, out)


@triton.jit
def rope_bw(
    FREQS, GRAD_OUT, GRAD_T,
    DIM, FREQS_DIM,
    BLOCK_SIZE: tl.constexpr
):
    # Compute index for this thread
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK_SIZE)
    seq_idx = pid // (DIM // BLOCK_SIZE)

    # Load gradient w.r.t. output
    grad_out = tl.load(GRAD_OUT + pid * BLOCK_SIZE + idx)
    # Load forward pass inputs
    freqs = tl.load(FREQS + seq_idx * FREQS_DIM + idx)
    freqs = tl.where(idx < FREQS_DIM, freqs, 0.0)

    # Calculate cos and sin again (as in forward)
    cos_ = tl.cos(freqs)
    sin_ = tl.sin(freqs)
    out_cos = grad_out * cos_

    # Backpropagate through rotate half logic
    half_dim = FREQS_DIM // 2
    new_idx = tl.where(idx >= half_dim, idx - half_dim, idx + half_dim)
    half_out = tl.load(GRAD_OUT + pid * BLOCK_SIZE + new_idx)
    rotated_out = tl.where(new_idx >= half_dim, half_out, -half_out)

    # Store the gradients
    grad_t = tl.where(idx < FREQS_DIM, out_cos + rotated_out * sin_, grad_out)
    tl.store(GRAD_T + pid * BLOCK_SIZE + idx, grad_t)


class TritonRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
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
        rope_fw[grid](
            t, freqs, out, 
            t.stride(0), freqs.stride(0),
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(freqs)

        return out.view(seq_len, b, h, d)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        freqs, = ctx.saved_tensors
        seq_len, b, h, d = grad_output.shape
        grad_output = grad_output.view(seq_len, -1).contiguous()
        freqs = freqs.view(seq_len, -1).contiguous()

        grad_t = torch.empty_like(grad_output)

        # Calculate 1D grid size
        BLOCK_SIZE = d
        total_tasks = seq_len * b * h * d
        grid = ((total_tasks + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        # Launch kernel with 1D grid
        rope_bw[grid](
            freqs, grad_output, grad_t,
            grad_output.stride(0), freqs.stride(0),
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return grad_t.view(seq_len, b, h, d), None, None

def apply_rotary_pos_emb_triton(
    t: torch.Tensor, 
    freqs: torch.Tensor
) -> torch.Tensor:
    return TritonRoPEFunc.apply(t, freqs)
