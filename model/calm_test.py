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

"""Tests for calm.py."""

import unittest

from model import calm
from model import utils
import torch


class CalmTest(unittest.TestCase):

  def setUp(self):
    """Sets up the CALM model for testing."""
    super().setUp()
    self.config = calm.CALMConfig(
        anchor_model="google/gemma-2b",
        aug_model="google/gemma-2b",
        num_connections=2,
        num_heads=1,
    )
    self.model = calm.CALM(self.config)

  def test_calm_config(self):
    """Tests that the CALM configuration is set correctly."""
    config = calm.CALMConfig(
        anchor_model="google/gemma-2b",
        aug_model="google/gemma-2b",
        num_connections=2,
        num_heads=1,
    )
    self.assertEqual(config.anchor_model, "google/gemma-2b")
    self.assertEqual(config.aug_model, "google/gemma-2b")
    self.assertEqual(config.num_connections, 2)

  def test_calm_forward(self):
    """Tests whether the CALM model returns the same output shape as the anchor model."""
    output = self.model(
        input_ids=torch.ones(1, 10),
        attention_mask=torch.ones(1, 10),
    )
    output_anchor_model = self.model.anchor_model(
        input_ids=torch.ones(1, 10),
        attention_mask=torch.ones(1, 10),
    )
    self.assertEqual(output[0].shape, output_anchor_model[0].shape)

  def test_calm_connections(self):
    """Tests that the CALM connections are set correctly."""
    config = calm.CALMConfig(
        anchor_model="google/gemma-2b",
        aug_model="google/gemma-2b",
        num_connections=2,
        num_heads=1,
    )
    model = calm.CALM(config)
    self.assertEqual(model.connections, [(0, 0), (17, 17)])

  def test_get_hidden_dim(self):
    """Tests that the hidden dimensions are set correctly."""
    for connection in self.model.connections:
      anchor_hidden_dim, aug_hidden_dim = utils.get_hidden_dims(
          self.model.anchor_model, self.model.aug_model, tuple(connection)
      )
      self.assertEqual(
          self.model.anchor_model.model.layers[connection[0]].hidden_size,
          anchor_hidden_dim
      )
      self.assertEqual(
          self.model.aug_model.model.layers[connection[1]].hidden_size,
          aug_hidden_dim
      )

  def test_cross_attention_hook(self):
    """Tests that the cross attention hook's embed_dim is same as anchor model's hidden size."""
    for connection_idx, connection in enumerate(self.model.connections):
      anchor_hidden_dim, _ = utils.get_hidden_dims(
          self.model.anchor_model, self.model.aug_model, tuple(connection)
      )
      self.assertEqual(
          self.model.cross_attention_hooks[
              connection_idx
          ].cross_attention.embed_dim,
          anchor_hidden_dim,
      )

  def test_calm_generate(self):
    """Tests if generate is working correctly."""
    input_ids = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=int)
    generate_ids = self.model.generate(input_ids, max_length=10)
    print(generate_ids)


if __name__ == "__main__":
  unittest.main()
