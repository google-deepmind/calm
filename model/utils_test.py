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

"""Tests for utils.py."""

import unittest

from model import calm
from model import utils


class UtilsTest(unittest.TestCase):

  def test_check_connections(self):
    """Tests if a few example connections are valid."""
    self.assertTrue(utils.check_connections([(0, 0), (1, 1)], 2, 2))
    self.assertFalse(utils.check_connections([(0, 0), (1, 1)], 1, 2))
    self.assertFalse(utils.check_connections([(0, 0), (1, 1)], 2, 1))

  def test_get_connections(self):
    """Tests that connections are formed correctly using get_connections."""
    self.assertEqual(utils.get_connections(2, 2, 2), [(0, 0), (1, 1)])
    self.assertEqual(utils.get_connections(1, 2, 2), [(0, 0)])
    self.assertEqual(utils.get_connections(2, 1, 2), [(0, 0), (0, 1)])
    self.assertEqual(utils.get_connections(2, 2, 1), [(0, 0), (1, 0)])

  def test_get_hidden_dims(self):
    """Tests that the hidden dimensions are set correctly."""
    config = calm.CALMConfig(
        anchor_model="google/gemma-2b",
        aug_model="google/gemma-2b",
        num_connections=2,
        num_heads=1,
    )
    model = calm.CALM(config)
    for connection in model.connections:
      anchor_hidden_dim, aug_hidden_dim = utils.get_hidden_dims(
          model.anchor_model, model.aug_model, tuple(connection)
      )
      self.assertEqual(
          model.anchor_model.model.layers[connection[0]].hidden_size,
          anchor_hidden_dim
      )
      self.assertEqual(
          model.aug_model.model.layers[connection[1]].hidden_size,
          aug_hidden_dim
      )

if __name__ == "__main__":
  unittest.main()
