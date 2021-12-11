.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

GPT-J
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPT-J model was released in the `kingoflolz/mesh-transformer-jax
<https://github.com/kingoflolz/mesh-transformer-jax>`__ repository by Ben Wang and Aran Komatsuzaki. It is a GPT-2-like
causal language model trained on `the Pile <https://pile.eleuther.ai/>`__ dataset.

This model was contributed by `Stella Biderman <https://huggingface.co/stellaathena>`__.

Tips:

- To load `GPT-J <https://huggingface.co/EleutherAI/gpt-j-6B>`__ in float32 one would need at least 2x model size CPU
  RAM: 1x for initial weights and another 1x to load the checkpoint. So for GPT-J it would take at least 48GB of CPU
  RAM to just load the model. To reduce the CPU RAM usage there are a few options. The ``torch_dtype`` argument can be
  used to initialize the model in half-precision. And the ``low_cpu_mem_usage`` argument can be used to keep the RAM
  usage to 1x. There is also a `fp16 branch <https://huggingface.co/EleutherAI/gpt-j-6B/tree/float16>`__ which stores
  the fp16 weights, which could be used to further minimize the RAM usage. Combining all this it should take roughly
  12.1GB of CPU RAM to load the model.

.. code-block::

    >>> from transformers import GPTJForCausalLM
    >>> import torch

    >>> model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)


- The model should fit on 16GB GPU for inference. For training/fine-tuning it would take much more GPU RAM. Adam
  optimizer for example makes four copies of the model: model, gradients, average and squared average of the gradients.
  So it would need at least 4x model size GPU memory, even with mixed precision as gradient updates are in fp32. This
  is not including the activations and data batches, which would again require some more GPU RAM. So one should explore
  solutions such as DeepSpeed, to train/fine-tune the model. Another option is to use the original codebase to
  train/fine-tune the model on TPU and then convert the model to Transformers format for inference. Instructions for
  that could be found `here <https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md>`__

- Although the embedding matrix has a size of 50400, only 50257 entries are used by the GPT-2 tokenizer. These extra
  tokens are added for the sake of efficiency on TPUs. To avoid the mis-match between embedding matrix size and vocab
  size, the tokenizer for `GPT-J <https://huggingface.co/EleutherAI/gpt-j-6B>`__ contains 143 extra tokens
  ``<|extratoken_1|>... <|extratoken_143|>``, so the ``vocab_size`` of tokenizer also becomes 50400.

Generation
_______________________________________________________________________________________________________________________

The :meth:`~transformers.generation_utils.GenerationMixin.generate` method can be used to generate text using GPT-J
model.

.. code-block::

    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    >>> prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
    ...          "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
    ...          "researchers was the fact that the unicorns spoke perfect English."

    >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    >>> gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
    >>> gen_text = tokenizer.batch_decode(gen_tokens)[0]

...or in float16 precision:

.. code-block::

    >>> from transformers import GPTJForCausalLM, AutoTokenizer
    >>> import torch

    >>> model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
    >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    >>> prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
    ...          "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
    ...          "researchers was the fact that the unicorns spoke perfect English."

    >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    >>> gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
    >>> gen_text = tokenizer.batch_decode(gen_tokens)[0]


GPTJConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTJConfig
    :members:

GPTJModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTJModel
    :members: forward


GPTJForCausalLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTJForCausalLM
    :members: forward


GPTJForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTJForSequenceClassification
    :members: forward


GPTJForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTJForQuestionAnswering
    :members: forward


FlaxGPTJModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaxGPTJModel
    :members: __call__


FlaxGPTJForCausalLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaxGPTJForCausalLM
    :members: __call__
