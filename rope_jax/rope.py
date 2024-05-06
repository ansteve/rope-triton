import jax
import jax.numpy as jnp
from flax import linen as nn 
from typing import Tuple


class RotaryEmbedding(nn.Module):
  def __init__(self, dim):
    # Calculate inv_freq in JAX
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    self.inv_freq = jax.device_put(inv_freq, jax.devices('gpu')[0])

  def __call__(self, max_seq_len: int, offset: int = 0):
    seq = jnp.arange(max_seq_len, dtype=self.inv_freq.dtype) + offset
    seq = jax.device_put(seq, jax.devices('gpu')[0])

    freqs = jnp.einsum('i , j -> i j', seq, self.inv_freq)
    # first part even vector components, second part odd vector components,
    #  2 * dim in dimension size
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    # emb [seq_length, .., dim]
    return emb.reshape((emb.shape[0], 1, 1, emb.shape[1]))


def _rotate_half(t: jnp.ndarray) -> jnp.ndarray:
  # Assuming t is a 3-dimensional tensor and rotation is applied along the last axis
  split_index = t.shape[-1] // 2
  return jnp.concatenate((t[..., split_index:], t[..., :split_index]), axis=-1)


def apply_rotary_emb(
    t: jnp.ndarray,
    freqs: jnp.ndarray,
    tensor_format: str = "sbhd",
) -> jnp.ndarray:
  assert tensor_format in ("sbhd", "bshd"), (
    "Only formats `sbhd` or `bshd` are supported for input tensor `t` "
    f"when fused is False, got {tensor_format}."
  )

  max_seq_len = freqs.shape[0]
  cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

  # Only apply the rotary embeddings up to the sequence length of the running
  # input.
  assert cur_seq_len <= max_seq_len, (
    f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
  )
  freqs = freqs[:cur_seq_len]
  # cos/sin first then dtype conversion for better precision
  cos_ = jnp.cos(freqs).astype(t.dtype)
  sin_ = jnp.sin(freqs).astype(t.dtype)

  rot_dim = freqs.shape[-1]
  # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
  t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

  # first part is cosine component
  # second part is sine component, need to change signs with _rotate_half method
  t = (t * cos_) + (_rotate_half(t) * sin_)
  return jnp.concatenate((t, t_pass), axis=-1)