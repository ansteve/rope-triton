import torch
import triton

from src.rope_torch import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)
from src.rope_triton import apply_rotary_pos_emb_triton

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seq_length'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(8, 12, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['torch', 'triton', 'cuda'],  # Possible values for `line_arg`.
        line_names=['Torch', 'Triton', 'Cuda'],  # Label name for the lines.
        styles=[('red', '-'), ('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        plot_name='rope-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(seq_length, provider):
    hidden_size = 128
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    t = torch.rand(
        (seq_length, batch_size, head_num, hidden_size),
        dtype=torch.float32,
        device=device,
    )
    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, 0.5)
    emb = rotary_pos_emb(seq_length)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms =  triton.testing.do_bench(
            lambda: apply_rotary_pos_emb(t, emb),
            quantiles=quantiles)
    if provider == 'cuda':
        ms, min_ms, max_ms =  triton.testing.do_bench(
            lambda: apply_rotary_pos_emb(t, emb, fused=True),
            quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: apply_rotary_pos_emb_triton(t, emb),
            quantiles=quantiles)
    return ms, max_ms, min_ms

benchmark.run(print_data=True, show_plots=True, save_path='./result')