---
base_model: gpt2
tags:
- generated_from_trainer
datasets:
- wikitext
metrics:
- accuracy
model-index:
- name: checkpoints
  results:
  - task:
      name: Causal Language Modeling
      type: text-generation
    dataset:
      name: wikitext wikitext-2-raw-v1
      type: wikitext
      config: wikitext-2-raw-v1
      split: validation
      args: wikitext-2-raw-v1
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.41395812968393614
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# checkpoints

This model is a fine-tuned version of [gpt2](https://huggingface.co/gpt2) on the wikitext wikitext-2-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 3.1272
- Accuracy: 0.4140

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1.0

### Training results



### Framework versions

- Transformers 4.35.2
- Pytorch 2.0.1
- Datasets 2.15.0
- Tokenizers 0.15.0
