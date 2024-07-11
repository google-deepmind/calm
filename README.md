
# CALM: Expanding LLM Capabilities through Composition

This repository provides the code for implementing the CALM (Composition to Augment Language Models) framework described in the paper "LLMs Augmented LLMs: Expanding Capabilities through Composition". In this paper, we describe composing two language models by introducing cross-attention between models to compose their representations and enable new capabilities. The code currently supports combining any two models built with the Gemma architecture.

## Installation

Clone the repo

```
git clone https://github.com/google-deepmind/calm.git
cd calm
```

Create a virtual environment using virtualenv or conda depending on your preferences and install the requirements. We require Python 3.10 or above:

```
conda create -n calm python=3.10 && conda activate calm
pip install -r requirements.txt
```

Ensure you have logged in using a 🤗 read access token for using the Gemma models. For more information, see: [🤗 User Access Tokens](https://huggingface.co/docs/hub/en/security-tokens).

```
huggingface-cli login
```

## Usage

Ex. Initialising a composed model

```
from model import calm

calm_config = calm.CALMConfig(
      anchor_model="google/gemma-2b",
      aug_model="google/gemma-2b",
      connections=[(0,0),(1,1)],  # Each element is a tuple (anchor_model_layer_index, aug_model_layer_index)
      num_heads=2,
)

model = calm.CALM(calm_config)
```
You can also use the `num_connections` argument to initialize the composed model, in which case connections are created uniformly across anchor and augmenting models.

```
calm_config = calm.CALMConfig(
      anchor_model="google/gemma-2b",
      aug_model="google/gemma-2b",
      num_connections=2,
      num_heads=2,
)
```

Ex. Saving and Loading a model

```
calm_config.save_pretrained('./calm_config')
model.save_pretrained('./calm')

config = CALMConfig.from_pretrained("./calm_config")
loaded_model = CALM.from_pretrained("./calm", config = config)
```

You can finetune the composed model using [🤗 Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)

```
training_args = TrainingArguments(
      output_dir="./tmp",
      overwrite_output_dir=True,
      num_train_epochs=epochs,
      do_train=True,
      do_eval=True,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      eval_strategy='steps',
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
      train_dataset=data['train'],
      eval_dataset=data['test'],
      data_collator=data_collator,
      tokenizer=tokenizer,
  )

trainer.can_return_loss = True

trainer.train()
```

An example multi-gpu training pipeline is given in `train.py` where we train a composed gemma-2b and gemma-7b using [Wikitext](https://huggingface.co/datasets/Salesforce/wikitext) data. You can run it using [🤗 Accelerate FSDP](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp)

An example accelerate config file is provided in `accelerate_config.yaml`

```
accelerate launch --config_file accelerate_config.yaml train.py \
          --anchor_model_dir google/gemma-7b \
          --aug_model_dir google/gemma-2b \
          --num_heads 2 \
          --num_connections 2 \
          --learning_rate 3e-4 \
          --batch_size  8 \
          --output_dir './tmp'
```

You can generate from the model the same way as any transformers model

```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
prompt = "I am going "
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(inputs.input_ids, max_length=10)
print(tokenizer.decode(generate_ids[0], skip_special_tokens=True))
```

## Citing this work


```latex
@misc{bansal2024llmaugmentedllmsexpanding,
      title={LLM Augmented LLMs: Expanding Capabilities through Composition},
      author={Rachit Bansal and Bidisha Samanta and Siddharth Dalmia and Nitish Gupta and
      Shikhar Vashishth and Sriram Ganapathy and Abhishek Bapna and Prateek Jain and Partha Talukdar},
      year={2024},
      eprint={2401.02412},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2401.02412},
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
