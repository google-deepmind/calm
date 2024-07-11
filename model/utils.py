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

"""Utils for CALM."""

import numpy as np


def check_connections(
    connections: list[tuple[int, int]],
    num_anchor_layers: int,
    num_aug_layers: int,
) -> bool:
  """Checks if the connections are valid."""
  for connection in connections:
    if connection[0] < 0 or connection[0] >= num_anchor_layers:
      print(
          f"Please verify your connections again. Index {connection[0]} doesn't"
          f" exist as anchor model only has {num_anchor_layers} layers"
      )
      return False
    if connection[1] < 0 or connection[1] >= num_aug_layers:
      print(
          f"Please verify your connections again. Index {connection[1]} doesn't"
          f" exist as augmenting model only has {num_aug_layers} layers"
      )
      return False
  return True


def get_connections(
    num_connections: int,
    num_anchor_layers: int,
    num_aug_layers: int,
) -> list[tuple[int, int]]:
  """Gets the connections for CALM."""
  anchor_layer = np.linspace(0, num_anchor_layers-1, num_connections, dtype=int)
  aug_layer = np.linspace(0, num_aug_layers-1, num_connections, dtype=int)

  return list(zip(anchor_layer, aug_layer))


def get_hidden_dims(
    anchor_model,
    aug_model,
    connection: tuple[int, int],
) -> tuple[int, int]:
  """Gets the hidden dimensions for the given layers."""
  anchor_layer, aug_layer = connection
  anchor_hidden_dim = anchor_model.model.layers[anchor_layer].hidden_size
  aug_hidden_dim = aug_model.model.layers[aug_layer].hidden_size

  return anchor_hidden_dim, aug_hidden_dim

