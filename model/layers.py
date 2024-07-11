# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Layer operation classes for CALM."""

from typing import Union

import torch
from transformers.models.gemma import modeling_gemma


def freeze_model(model):
  """Freezes the model."""
  for param in model.parameters():
    param.requires_grad = False


def process_hook_args(
    model: torch.nn.Module,  # pylint: disable=unused-argument
    inp: Union[torch.Tensor, tuple[torch.Tensor, ...]],  # pylint: disable=unused-argument
    out: Union[torch.Tensor, tuple[torch.Tensor, ...]],
):
  """Extracts the main output tensor from a PyTorch hook output.

  Args:
      model: The nn.Module object to which the hook is attached.
      inp: Input tensor to the layer (ignored).
      out: Output from the layer. This can be a tensor or a tuple containing the
        tensor.
  Reference:
    register_forward_hook in
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html
  Returns:
      The main output tensor from the hooked block.
  """
  anchor_hidden_state = out[0] if isinstance(out, tuple) else out
  query = anchor_hidden_state
  return query, out


class CrossAttentionHook(torch.nn.Module):
  """cross attention hook for CALM."""

  def __init__(
      self,
      anchor_hidden_dim: int,
      aug_hidden_dim: int,
      num_heads: int,
      rms_norm_eps: float = 1e-6,
  ):
    """Initializes the cross attention hook.

    Args:
      anchor_hidden_dim: The hidden dimension of the anchor model.
      aug_hidden_dim: The hidden dimension of the augmented model.
      num_heads: The number of attention heads in the hook
      rms_norm_eps: The epsilon value for the post-attention RMS norm layer

    Attributes:
      proj: The projection layer to project the augmented hidden state to the
        anchor hidden dimension.
      embed_dim: The hidden dimension of the anchor model.
      num_heads: The number of attention heads in the hook.
      cross_attention: The cross attention layer.
      aug_hidden_state: The augmented hidden state tensor. This is set by
        forward_aug in CALM.
      aug_mask: The augmented mask tensor. This is set by forward_aug in CALM.
      attn_weights: The attention weights tensor. This is set by the forward
        pass of the cross attention hook.
    Example:
      hook = CrossAttentionHook(anchor_hidden_dim, aug_hidden_dim, num_heads)
      model.register_forward_hook(hook)
      model(input)
      print(hook.attn_weights)
    """
    super().__init__()
    self.proj = torch.nn.Linear(aug_hidden_dim, anchor_hidden_dim)
    self.embed_dim = anchor_hidden_dim
    self.num_heads = num_heads
    self.post_attention_layernorm = modeling_gemma.GemmaRMSNorm(
        self.embed_dim, eps=rms_norm_eps
    )
    self.cross_attention = torch.nn.MultiheadAttention(
        self.embed_dim,
        num_heads,
        kdim=self.embed_dim,
        vdim=self.embed_dim,
        batch_first=True,
    )
    self.aug_hidden_state = None
    self.aug_mask = None
    self.attn_weights = None

  def forward(self, *hook_args):
    """Forward pass of the cross attention hook.

    Args:
      *hook_args: The arguments passed to the hook.

    Raises:
      ValueError: If aug_hidden_state or aug_mask is None.

    The cross attention hook is registered to the anchor model. The hook
    extracts the hidden state from the anchor model and uses it as the query
    for the cross attention. The key and value for the cross attention are
    computed by projecting the hidden state from the augmented model. The
    augmented hidden state and mask are set by forward_aug in CALM.

    Returns:
      The modified output of the cross attention hook.
    """
    query, output = process_hook_args(*hook_args)
    assert self.aug_hidden_state is not None
    assert self.aug_mask is not None
    key = self.proj(self.aug_hidden_state)
    value = self.proj(self.aug_hidden_state)

    self.aug_mask = self.aug_mask.float()
    attn_output, attn_weights = self.cross_attention(
        query, key, value, need_weights=True
    )
    self.attn_weights = attn_weights

    attn_output = self.post_attention_layernorm(attn_output)
    output_fin = attn_output + query
    new_output = (output_fin,) + output[1:]
    return new_output


class ExtractHiddenStateHook(torch.nn.Module):
  """Extract hidden state hook for CALM."""

  def __init__(self):
    """Initializes the extract hidden state hook.

    Attributes:
      hidden_state: The hidden state tensor. This is set by the forward pass of
        the extract hidden state hook.
    Example:
    ```python
      hook = ExtractHiddenStateHook()
      model.register_forward_hook(hook)
      model(input)
      print(hook.hidden_state)
    ```
    """
    super().__init__()
    self.hidden_state = None

  def forward(self, *hook_args):
    hidden_state, out = process_hook_args(*hook_args)
    self.hidden_state = hidden_state
    return out


