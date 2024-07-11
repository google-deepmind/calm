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

"""CALM training script for finetuning.

Hugging Face Trainer is used to train the CALM model.
Reference:
https://huggingface.co/docs/transformers/main_classes/trainer
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
import datasets
from model import calm
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments


_ANCHOR_MODEL_DIR = flags.DEFINE_string(
    'anchor_model_dir', None, 'anchor model path.'
)
_AUG_MODEL_DIR = flags.DEFINE_string('aug_model_dir', None, 'aug model path.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'output directory.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 2e-5, 'learning rate.')
_EPOCHS = flags.DEFINE_integer('epochs', 3, 'number of epochs.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 1, 'batch size.')
_NUM_HEADS = flags.DEFINE_integer('num_heads', 1, 'number of heads.')
_NUM_CONNECTIONS = flags.DEFINE_integer(
    'num_connections', 2, 'number of connections.'
)
_CONNECTIONS = flags.DEFINE_list(
    'connections',
    None,
    'connections between the anchor and aug model. You cannot provide both'
    'connections and num_connections simultaneously.',
)
_EVAL_STEPS = flags.DEFINE_integer('eval_steps', 50, 'eval steps.')
_LOGGING_STEPS = flags.DEFINE_integer('logging_steps', 50, 'logging steps.')
_SAVE_STEPS = flags.DEFINE_integer('save_steps', 50, 'save steps.')
_MAX_STEPS = flags.DEFINE_integer('max_steps', 100, 'max steps.')


def train(argv: Sequence[str]) -> None:
  """Trains the CALM model."""
  del argv  # Unused.
  anchor_model_path = _ANCHOR_MODEL_DIR.value
  aug_model_path = _AUG_MODEL_DIR.value
  num_heads = _NUM_HEADS.value
  num_connections = _NUM_CONNECTIONS.value
  logging.info('anchor_model_path: %s', anchor_model_path)
  logging.info('aug_model_path: %s', aug_model_path)
  logging.info('Loading Tokenizer...')
  tokenizer = AutoTokenizer.from_pretrained(anchor_model_path)
  logging.info('Loading Composed Model...')
  calm_config = calm.CALMConfig(
      anchor_model=anchor_model_path,
      aug_model=aug_model_path,
      anchor_config=None,
      aug_config=None,
      num_connections=num_connections,
      num_heads=num_heads,
  )

  model = calm.CALM(calm_config)
  train_data = datasets.load_dataset(
      path='Salesforce/wikitext', name='wikitext-2-raw-v1'
  )

  def preprocess_function(examples):
    return tokenizer(
        examples['text'], truncation=True, padding='max_length', max_length=512
    )

  train_data = train_data.map(preprocess_function, batched=True)
  data_collator = DataCollatorForLanguageModeling(
      tokenizer=tokenizer, mlm=False
  )

  epochs = _EPOCHS.value
  batch_size = _BATCH_SIZE.value
  learning_rate = _LEARNING_RATE.value
  output_dir = _OUTPUT_DIR.value
  eval_steps = _EVAL_STEPS.value
  logging_steps = _LOGGING_STEPS.value
  save_steps = _SAVE_STEPS.value
  max_steps = _MAX_STEPS.value
  training_args = TrainingArguments(
      output_dir=output_dir,
      overwrite_output_dir=True,
      num_train_epochs=epochs,
      do_train=True,
      do_eval=True,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      eval_strategy='steps',  # pylint:disable=unexpected-keyword-arg
      eval_steps=eval_steps,
      logging_steps=logging_steps,
      save_steps=save_steps,
      max_steps=max_steps,
      learning_rate=learning_rate,
      label_names=[],
      report_to=['tensorboard'],
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_data['train'],
      eval_dataset=train_data['test'],
      data_collator=data_collator,
      tokenizer=tokenizer,
  )

  trainer.can_return_loss = True

  trainer.train()

  trainer.save_model(
      output_dir,
  )

  print(f'Training complete! Model saved to {output_dir}')


if __name__ == '__main__':
  app.run(train)
