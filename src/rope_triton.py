import torch
import triton
import triton.language as tl


@triton.jit
def rope_fw(
    T, FREQS, OUT, 
    BH, FREQS_DIM,
    BLOCK_SIZE: tl.constexpr
):
    # Compute index for this thread
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK_SIZE)
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


@triton.jit
def rope_bw(
    FREQS, GRAD_OUT, GRAD_T,
    BH, FREQS_DIM,
    BLOCK_SIZE: tl.constexpr
):
    # Compute index for this thread
    pid = tl.program_id(0)
    idx = tl.arange(0, BLOCK_SIZE)
    seq_idx = pid // BH

    # Load gradient w.r.t. output
    grad_out = tl.load(GRAD_OUT + pid * BLOCK_SIZE + idx)
    # Load forward pass inputs
    freqs = tl.load(FREQS + seq_idx * FREQS_DIM + idx)
    freqs = tl.where(idx < FREQS_DIM, freqs, 0.0)

    # Calculate cos and sin again (as in forward)
    cos_ = tl.cos(freqs)
    sin_ = tl.sin(freqs)
    t_cos = grad_out * cos_

    # Backpropagate through rotate half logic
    half_dim = FREQS_DIM // 2
    new_idx = tl.where(idx >= half_dim, idx - half_dim, idx + half_dim)
    half_t = tl.load(GRAD_OUT + pid * BLOCK_SIZE + new_idx)
    rotated_t = tl.where(new_idx >= half_dim, half_t, -half_t)

    # Store the gradients
    res = tl.where(idx < FREQS_DIM, t_cos + rotated_t * sin_, grad_out)
    tl.store(GRAD_T + pid * BLOCK_SIZE + idx, res)


def _rotate_half2(x: torch.Tensor) -> torch.Tensor:
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((x2, -x1), dim=-1)


class TritonRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        # Reshape and pad input tensor and freqs
        seq_len, b, h, d = t.shape
        #t = t.view(seq_len, -1).contiguous()
        #freqs = freqs.view(seq_len, -1).contiguous()

        # Allocate output tensor
        out = torch.empty_like(t)

        # Calculate 1D grid size
        BLOCK_SIZE = d
        total_tasks = seq_len * b * h * d
        grid = ((total_tasks + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        # Launch kernel with 1D grid
        rope_fw[grid](
            t, freqs, out, 
            b * h, freqs.shape[-1],
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(freqs)

        return out.view(seq_len, b, h, d)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        freqs, = ctx.saved_tensors
        seq_len, b, h, d = grad_output.shape

        # cos_ = torch.cos(freqs).to(t.dtype)
        # sin_ = torch.sin(freqs).to(t.dtype)
        # freqs_dim = freqs.shape[-1]

        # # grad_output으로부터 그래디언트 계산 시작
        # grad_t = grad_output[..., :freqs_dim]
        # grad_t_pass = grad_output[..., freqs_dim:]

        # # Calculate gradients for rotated part
        # grad_original_t = (grad_t * cos_) + (_rotate_half2(grad_t) * sin_)

        # # Concatenate the gradients for t
        # grad_t_total = torch.cat((grad_original_t, grad_t_pass), dim=-1)

        grad_t = torch.empty_like(grad_output)

        # Calculate 1D grid size
        BLOCK_SIZE = d
        total_tasks = seq_len * b * h * d
        grid = ((total_tasks + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        # Launch kernel with 1D grid
        rope_bw[grid](
            freqs, grad_output, grad_t,
            b * h, freqs.shape[-1],
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return grad_t, None, None

def apply_rotary_pos_emb_triton(
    t: torch.Tensor, 
    freqs: torch.Tensor
) -> torch.Tensor:
    return TritonRoPEFunc.apply(t, freqs)
