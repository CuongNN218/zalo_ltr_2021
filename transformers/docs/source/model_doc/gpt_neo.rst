.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

GPT Neo
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GPTNeo model was released in the `EleutherAI/gpt-neo <https://github.com/EleutherAI/gpt-neo>`__ repository by Sid
Black, Stella Biderman, Leo Gao, Phil Wang and Connor Leahy. It is a GPT2 like causal language model trained on the
`Pile <https://pile.eleuther.ai/>`__ dataset.

The architecture is similar to GPT2 except that GPT Neo uses local attention in every other layer with a window size of
256 tokens.

This model was contributed by `valhalla <https://huggingface.co/valhalla>`__.

Generation
_______________________________________________________________________________________________________________________

The :obj:`generate()` method can be used to generate text using GPT Neo model.

.. code-block::

    >>> from transformers import GPTNeoForCausalLM, GPT2Tokenizer
    >>> model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    >>> tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

    >>> prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
    ...          "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
    ...          "researchers was the fact that the unicorns spoke perfect English."

    >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    >>> gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
    >>> gen_text = tokenizer.batch_decode(gen_tokens)[0]


GPTNeoConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTNeoConfig
    :members:


GPTNeoModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTNeoModel
    :members: forward


GPTNeoForCausalLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTNeoForCausalLM
    :members: forward

GPTNeoForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GPTNeoForSequenceClassification
    :members: forward

FlaxGPTNeoModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaxGPTNeoModel
    :members: __call__


FlaxGPTNeoForCausalLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaxGPTNeoForCausalLM
    :members: __call__
