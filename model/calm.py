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

"""CALM implementation."""

import os
from typing import Callable, List, Optional, Tuple, Union
from model import layers
from model import utils
import torch
import transformers


class CALMConfig(transformers.PretrainedConfig):
  """CALM configuration.

  Configuration file for CALM. Enables the user to specify the anchor and
  augmented models, the number of connections, and the number of heads in each
  cross attention hook.
  """
  model_type = "calm"

  def __init__(
      self,
      anchor_model: str = "google/gemma-2b",
      aug_model: str = "google/gemma-2b",
      anchor_config: Optional[transformers.AutoConfig] = None,
      aug_config: Optional[transformers.AutoConfig] = None,
      connections: list[Tuple[int, int]] = None,
      num_connections: int = None,
      num_heads: int = 1,
      **kwargs,
  ):
    """CALM configuration.

    Args:
      anchor_model: HF Repo ID or Path to the anchor model.
      aug_model: HF Repo ID or Path to the augmented model.
      anchor_config: Config for the anchor model. If None, the config will be
        loaded from the anchor model. If a dict is provided, it will be
        converted to a GemmaConfig.
      aug_config: Config for the augmenting model. If None, the config will be
        loaded from the augmenting model. If a dict is provided, it will be
        converted to a GemmaConfig.
      connections: The connections between the anchor and augmented models. If
        None, num_connections must be set. Every connection is a tuple of
        (anchor_layer_idx, aug_layer_idx).
      num_connections: The number of connections between the anchor and
        augmented models. If none, connections must be set.
      num_heads: The number of attention heads in each cross attention hook.
      **kwargs:
    """

    self.anchor_model = anchor_model
    self.aug_model = aug_model
    self.connections = connections
    self.num_connections = num_connections
    self.num_heads = num_heads
    self.anchor_config = anchor_config
    self.aug_config = aug_config
    super().__init__(**kwargs)


class CALM(transformers.PreTrainedModel):
  """CALM implementation.

  Class for composing the anchor and augmented models. The class is designed to
  integrate with the transformers library. You can use the CALM object for
  training, evaluation, and inference just like any other transformers model.
  """

  config_class = CALMConfig

  @property
  def lm_head(self):
    """Returns the language model head."""
    return self.anchor_model.lm_head

  def __init__(self, config: CALMConfig):
    """CALM implementation.

    Args:
      config: CALMConfig.

    Raises:
      ValueError: If config.connections is None and config.num_connections is
        None.

    Initializes the CALM model by composing the anchor and augmented models.
    The anchor model and the augmenting model are frozen and the augmented model
    is used to provide hidden states for the cross attention hooks. The
    cross attention hooks are registered to the anchor model.
    """
    super().__init__(config)  # pylint: disable=too-many-function-args
    if config.anchor_config is None:
      config.anchor_config = transformers.AutoConfig.from_pretrained(
          config.anchor_model
      )
    if config.aug_config is None:
      config.aug_config = transformers.AutoConfig.from_pretrained(
          config.aug_model
      )
    if isinstance(config.anchor_config, dict):
      config.anchor_config = transformers.GemmaConfig.from_dict(
          config.anchor_config
      )
    if isinstance(config.aug_config, dict):
      config.aug_config = transformers.GemmaConfig.from_dict(config.aug_config)

    self.anchor_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.anchor_model,
        config=config.anchor_config,
    )
    self.aug_model = transformers.AutoModelForCausalLM.from_pretrained(
        config.aug_model,
        config=config.aug_config,
    )
    self.vocab_size = self.anchor_model.config.vocab_size
    self.config = config
    self.num_anchor_layers = len(self.anchor_model.model.layers)
    self.num_aug_layers = len(self.aug_model.model.layers)

    assert (config.connections is None) ^ (config.num_connections is None)

    if config.connections is not None:
      assert utils.check_connections(
          config.connections, self.num_anchor_layers, self.num_aug_layers
      )
      self.connections = config.connections
      self.num_connections = len(config.connections)
    else:
      self.num_connections = config.num_connections
      self.connections = utils.get_connections(
          config.num_connections, self.num_anchor_layers, self.num_aug_layers
      )

    self.extract_hidden_state_hooks = {}
    for connection in self.connections:
      aug_connection_idx = connection[1]
      hook = layers.ExtractHiddenStateHook()
      self.extract_hidden_state_hooks[tuple(connection)] = hook
      self.aug_model.model.layers[aug_connection_idx].register_forward_hook(
          hook
      )

    self.connection_hidden_dims = []
    for connection in self.connections:
      anchor_hidden_dim, aug_hidden_dim = utils.get_hidden_dims(
          self.anchor_model, self.aug_model, tuple(connection)
      )
      self.connection_hidden_dims.append((anchor_hidden_dim, aug_hidden_dim))

    self.cross_attention_hooks = torch.nn.ModuleList([])

    for _, connection_hidden_dim in zip(
        self.connections, self.connection_hidden_dims
    ):
      self.cross_attention_hooks.append(
          layers.CrossAttentionHook(
              anchor_hidden_dim=connection_hidden_dim[0],
              aug_hidden_dim=connection_hidden_dim[1],
              num_heads=config.num_heads,
              rms_norm_eps=self.anchor_model.config.rms_norm_eps,
          )
      )

    layers.freeze_model(self.anchor_model)
    layers.freeze_model(self.aug_model)

    for connection_idx, connection in enumerate(self.connections):
      connection_anchor_layer_idx = connection[0]
      layer = self.anchor_model.model.layers[connection_anchor_layer_idx]
      layer.register_forward_hook(self.cross_attention_hooks[connection_idx])

  def release_memory(self):
    """Frees the memory of the CALM model after every forward pass."""

    for cross_attention_hook in self.cross_attention_hooks:
      cross_attention_hook.aug_hidden_state = None
      cross_attention_hook.aug_mask = None
      cross_attention_hook.attn_weights = None
    for extract_hidden_state_hook in self.extract_hidden_state_hooks.values():
      extract_hidden_state_hook.hidden_state = None

  def _forward_aug(
      self,
      input_ids: torch.LongTensor = None,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.LongTensor] = None,
      past_key_values: Optional[
          Union[transformers.Cache, List[torch.FloatTensor]]
      ] = None,
      inputs_embeds: Optional[torch.FloatTensor] = None,
      labels: Optional[torch.LongTensor] = None,
      use_cache: Optional[bool] = True,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
      cache_position: Optional[torch.LongTensor] = None,
  ):
    """Forwards the sequence through the augmented model.

    Args:
      input_ids: Input sequence.
      attention_mask: Input sequence mask.
      position_ids: Position ids.
      past_key_values: Past key values.
      inputs_embeds: Input embeddings.
      labels: Labels. If None, the model will be used in inference mode. If
        labels are provided, the model will be used in training mode.
      use_cache: Use cache.
      output_attentions: Output attentions.
      output_hidden_states: Output hidden states.
      return_dict: Return dict. If True, the output will be a dict. If False,
        the output will be a tuple.
      cache_position: Cache position.

    Returns:
      output: Output of the augmented model.

    The intermediate hidden states are extracted and are used to provide hidden
    states for the cross attention hooks. The output of the augmented model is
    returned.
    """

    with torch.no_grad():
      self.aug_model.eval()
      output = self.aug_model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          position_ids=position_ids,
          past_key_values=past_key_values,
          inputs_embeds=inputs_embeds,
          labels=labels,
          use_cache=use_cache,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
          cache_position=cache_position,
      )
      for connection_idx, connection in enumerate(self.connections):
        aug_hidden_state = self.extract_hidden_state_hooks[
            tuple(connection)
        ].hidden_state
        self.cross_attention_hooks[connection_idx].aug_hidden_state = (
            aug_hidden_state
        )
        self.cross_attention_hooks[connection_idx].aug_mask = attention_mask
        del aug_hidden_state
    return output

  def forward(
      self,
      input_ids: torch.LongTensor = None,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.LongTensor] = None,
      past_key_values: Optional[
          Union[transformers.Cache, List[torch.FloatTensor]]
      ] = None,
      inputs_embeds: Optional[torch.FloatTensor] = None,
      labels: Optional[torch.LongTensor] = None,
      use_cache: Optional[bool] = True,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
      cache_position: Optional[torch.LongTensor] = None,
  ):
    """CALM forward pass.

    Args:
      input_ids: Input sequence.
      attention_mask: Input sequence mask.
      position_ids: Position ids.
      past_key_values: Past key values.
      inputs_embeds: Input embeddings.
      labels: Labels. If None, the model will be used in inference mode. If
        labels are provided, the model will be used in training mode.
      use_cache: Use cache.
      output_attentions: Output attentions.
      output_hidden_states: Output hidden states.
      return_dict: Return dict. If True, the output will be a dict. If False,
        the output will be a tuple.
      cache_position: Cache position.

    Returns:
      The output of the CALM model. If labels are provided, the output
      will be the loss. If labels are not provided, the output will be the
      same class of the anchor model's output.

    Example:
      config = CALMConfig(
        anchor_model='google/gemma-2b',
        aug_model='google/gemma-2b',
        num_connections=2,
        num_heads=1,
      )
      model = CALM(config)
      output = model(input_ids, attention_mask)
    """
    aug_output = self._forward_aug(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        )
    del aug_output

    output = self.anchor_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    return output

  def save_pretrained(
      self,
      save_directory: Union[str, os.PathLike[str]],
      is_main_process: bool = True,
      state_dict: Optional[
          dict[str, dict[str, dict[str, torch.Tensor]]]
      ] = None,
      save_function: Callable[..., None] = torch.save,
      push_to_hub: bool = False,
      max_shard_size: Union[int, str] = "10GB",
      safe_serialization: bool = True,
      variant: Optional[str] = None,
      token: Optional[Union[bool, str]] = None,
      save_peft_format: bool = False,
      **kwargs,
  ):
    """Save the CALM model to a directory.

    Args:
      save_directory: The directory to save the model to.
      is_main_process: Whether this process is the main process.
      state_dict: The state dictionary to save.
      save_function: The function to use to save the state dictionary.
      push_to_hub: Whether to push the model to the hub.
      max_shard_size: The maximum shard size.
      safe_serialization: Whether to allow safe serialization. Set false to
        allow saving models with shared weights.
      variant: The variant of the model to save.
      token: The token to use to push the model to the hub.
      save_peft_format: Whether to save the model in the PEFT format.
      **kwargs: Additional keyword arguments.

    This method overrides the default save_pretrained method to handle the
    shared weights issue. It sets safe_serialization to False to allow saving
    models with shared weights.
    """
    super().save_pretrained(  # pytype: disable=attribute-error
        save_directory=save_directory,
        is_main_process=is_main_process,
        state_dict=state_dict,
        save_function=save_function,
        push_to_hub=push_to_hub,
        max_shard_size=max_shard_size,
        safe_serialization=False,
        variant=variant,
        token=token,
        save_peft_format=save_peft_format,
        **kwargs,
    )

  def prepare_inputs_for_generation(
      self,
      input_ids,
      past_key_values=None,
      attention_mask=None,
      inputs_embeds=None,
      cache_position=None,
      use_cache=True,
      **kwargs,
  ):
    """Prepares the inputs for generation.

    Args:
      input_ids: Input sequence.
      past_key_values: Past key values.
      attention_mask: Input sequence mask.
      inputs_embeds: Input embeddings.
      cache_position: Cache position.
      use_cache: Use cache.
      **kwargs: Additional keyword arguments.

    Returns:
      The prepared inputs for generation.
    """
    past_length = 0
    if past_key_values is not None:
      if isinstance(past_key_values, transformers.Cache):
        past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()  # pylint: disable=line-too-long
        max_cache_length = (
            torch.tensor(past_key_values.get_max_length(), device=input_ids.device)  # pylint: disable=line-too-long
            if past_key_values.get_max_length() is not None
            else None
        )
        cache_length = (
            past_length
            if max_cache_length is None
            else torch.min(max_cache_length, past_length)
        )
      else:
        cache_length = past_length = past_key_values[0][0].shape[2]
        max_cache_length = None

      # Keep only the unprocessed tokens:
      # 1 - If the length of the attention_mask exceeds the length of
      # input_ids, then we are in a setting where some of the inputs are
      # exclusively passed as part of the cache (e.g. when passing
      # input_embeds as input)
      if (
          attention_mask is not None
          and attention_mask.shape[1] > input_ids.shape[1]
      ):
        input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
      # 2 - If the past_length is smaller than input_ids.shape[1], then
      # input_ids holds all input tokens. We can discard input_ids based on
      # the past_length.
      elif past_length < input_ids.shape[1]:
        input_ids = input_ids[:, past_length:]
      # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume
      # input_ids only has unprocessed tokens.

      # If we are about to go beyond the maximum cache length, we need to crop
      # the input attention mask.
      if (
          max_cache_length is not None
          and attention_mask is not None
          and cache_length + input_ids.shape[1] > max_cache_length
      ):
        attention_mask = attention_mask[:, -max_cache_length :]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
      # create position_ids on the fly for batch generation
      position_ids = attention_mask.long().cumsum(-1) - 1
      position_ids.masked_fill_(attention_mask == 0, 1)
      if past_key_values:
        position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st
    # generation step
    if inputs_embeds is not None and past_key_values is None:
      model_inputs = {"inputs_embeds": inputs_embeds.contiguous()}
    else:
      model_inputs = {"input_ids": input_ids.contiguous()}

    input_length = (
        position_ids.shape[-1]
        if position_ids is not None
        else input_ids.shape[-1]
    )
    if cache_position is None:
      cache_position = torch.arange(
          past_length, past_length + input_length, device=input_ids.device
      )
    elif use_cache:
      cache_position = cache_position[-input_length:]

    model_inputs.update({
        "position_ids": position_ids,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "attention_mask": attention_mask,
    })

    return model_inputs
