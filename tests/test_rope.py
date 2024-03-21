import pytest
import torch
from typing import Callable, Dict, Tuple, Union
from src.rope_torch import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)
from src.rope_triton import apply_rotary_pos_emb_triton


def get_tol(dtype: torch.dtype) -> Dict:
    if dtype == torch.bfloat16:
        return dict(atol=1e-2, rtol=1e-2)
    elif dtype == torch.float16:
        return dict(atol=1e-3, rtol=1e-3)
    return dict(atol=1e-5, rtol=1.3e-6)


# Gradient is a broadcasted scalar
def _overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    return output.sum() * 2

# Gradient is a full tensor
def _non_overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    t = torch.ones_like(output)
    return torch.sum(output * t)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("seq_length", [2048, 4096])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
@pytest.mark.parametrize("margin", [0])
@pytest.mark.parametrize("transpose", [None])
@pytest.mark.parametrize("tensor_format", ["sbhd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_fused_rope(
    dtype: torch.dtype,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    margin: int,
    transpose: Union[Tuple, None],
    tensor_format: str,
    loss_func: Callable,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    t = torch.rand(
        (seq_length - margin, batch_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    if tensor_format == "bshd":
        t = t.transpose(0, 1).contiguous()
    if transpose:
        t = t.transpose(*transpose).contiguous().transpose(*transpose)
    t.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(seq_length)

    # triton
    output_triton = apply_rotary_pos_emb_triton(
        t, emb
    )
    # print('output_triton', output_triton)
    loss_triton = loss_func(output_triton)
    loss_triton.backward()
    grad_triton = t.grad.detach().clone()
    t.grad = None

    # pytorch
    output_torch = apply_rotary_pos_emb(
        t,
        emb,
        tensor_format=tensor_format,
        fused=False,
    )
    # print('output_torch', output_torch)
    loss_torch = loss_func(output_torch)
    loss_torch.backward()
    grad_torch = t.grad.detach().clone()
    t.grad = None

    torch.testing.assert_close(output_triton, output_torch, **get_tol(dtype))
    #torch.testing.assert_close(grad_triton, grad_torch, **get_tol(dtype))
    assert output_triton.is_contiguous()
