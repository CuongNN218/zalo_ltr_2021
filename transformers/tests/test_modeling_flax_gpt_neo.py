# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tempfile
import unittest

import numpy as np

import transformers
from transformers import GPT2Tokenizer, GPTNeoConfig, is_flax_available, is_torch_available
from transformers.testing_utils import is_pt_flax_cross_test, require_flax, slow

from .test_generation_flax_utils import FlaxGenerationTesterMixin
from .test_modeling_flax_common import FlaxModelTesterMixin, ids_tensor, random_attention_mask


if is_flax_available():
    import jax
    import jax.numpy as jnp
    from transformers.modeling_flax_pytorch_utils import (
        convert_pytorch_state_dict_to_flax,
        load_flax_weights_in_pytorch_model,
    )
    from transformers.models.gpt_neo.modeling_flax_gpt_neo import FlaxGPTNeoForCausalLM, FlaxGPTNeoModel

if is_torch_available():
    import torch


class FlaxGPTNeoModelTester:
    def __init__(
        self,
        parent,
        batch_size=14,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        attention_types=[[["global", "local"], 2]],
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        window_size=7,
        initializer_range=0.02,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_types = attention_types
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.window_size = window_size
        self.initializer_range = initializer_range
        self.scope = None
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = GPTNeoConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=False,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            window_size=self.window_size,
            attention_types=self.attention_types,
        )

        return (config, input_ids, input_mask)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict

    def check_use_cache_forward(self, model_class_name, config, input_ids, attention_mask):
        max_decoder_length = 20
        model = model_class_name(config)

        past_key_values = model.init_cache(input_ids.shape[0], max_decoder_length)
        attention_mask = jnp.ones((input_ids.shape[0], max_decoder_length), dtype="i4")

        position_ids = jnp.broadcast_to(
            jnp.arange(input_ids.shape[-1] - 1)[None, :], (input_ids.shape[0], input_ids.shape[-1] - 1)
        )
        outputs_cache = model(
            input_ids[:, :-1],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        position_ids = jnp.array(input_ids.shape[0] * [[input_ids.shape[-1] - 1]], dtype="i4")
        outputs_cache_next = model(
            input_ids[:, -1:],
            attention_mask=attention_mask,
            past_key_values=outputs_cache.past_key_values,
            position_ids=position_ids,
        )

        outputs = model(input_ids)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")

    def check_use_cache_forward_with_attn_mask(self, model_class_name, config, input_ids, attention_mask):
        max_decoder_length = 20
        model = model_class_name(config)

        attention_mask_cache = jnp.concatenate(
            [attention_mask, jnp.zeros((attention_mask.shape[0], max_decoder_length - attention_mask.shape[1]))],
            axis=-1,
        )

        past_key_values = model.init_cache(input_ids.shape[0], max_decoder_length)
        position_ids = jnp.broadcast_to(
            jnp.arange(input_ids.shape[-1] - 1)[None, :], (input_ids.shape[0], input_ids.shape[-1] - 1)
        )

        outputs_cache = model(
            input_ids[:, :-1],
            attention_mask=attention_mask_cache,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_ids = jnp.array(input_ids.shape[0] * [[input_ids.shape[-1] - 1]], dtype="i4")
        outputs_cache_next = model(
            input_ids[:, -1:],
            past_key_values=outputs_cache.past_key_values,
            attention_mask=attention_mask_cache,
            position_ids=position_ids,
        )

        outputs = model(input_ids, attention_mask=attention_mask)

        diff = np.max(np.abs((outputs_cache_next[0][:, -1, :5] - outputs[0][:, -1, :5])))
        self.parent.assertTrue(diff < 1e-3, msg=f"Max diff is {diff}")


@require_flax
class FlaxGPTNeoModelTest(FlaxModelTesterMixin, FlaxGenerationTesterMixin, unittest.TestCase):

    all_model_classes = (FlaxGPTNeoModel, FlaxGPTNeoForCausalLM) if is_flax_available() else ()
    all_generative_model_classes = (FlaxGPTNeoForCausalLM,) if is_flax_available() else ()

    def setUp(self):
        self.model_tester = FlaxGPTNeoModelTester(self)

    def test_use_cache_forward(self):
        for model_class_name in self.all_model_classes:
            config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
            self.model_tester.check_use_cache_forward(model_class_name, config, input_ids, attention_mask)

    def test_use_cache_forward_with_attn_mask(self):
        for model_class_name in self.all_model_classes:
            config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
            self.model_tester.check_use_cache_forward_with_attn_mask(
                model_class_name, config, input_ids, attention_mask
            )

    @slow
    def test_batch_generation(self):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token="<|endoftext|>", padding_side="left")
        inputs = tokenizer(["Hello this is a long string", "Hey"], return_tensors="np", padding=True, truncation=True)

        model = FlaxGPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
        model.do_sample = False
        model.config.pad_token_id = model.config.eos_token_id

        jit_generate = jax.jit(model.generate)

        output_sequences = jit_generate(
            inputs["input_ids"], attention_mask=inputs["attention_mask"], pad_token_id=tokenizer.pad_token_id
        ).sequences

        output_string = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        expected_string = [
            "Hello this is a long string of text.\n\nI'm trying to get the text of the",
            "Hey, I'm a little late to the party. I'm going to",
        ]

        self.assertListEqual(output_string, expected_string)

    # overwrite from common since `attention_mask` in combination
    # with `causal_mask` behaves slighly differently
    @is_pt_flax_cross_test
    def test_equivalence_pt_to_flax(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                # prepare inputs
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                pt_inputs = {k: torch.tensor(v.tolist()) for k, v in prepared_inputs_dict.items()}

                # load corresponding PyTorch class
                pt_model_class_name = model_class.__name__[4:]  # Skip the "Flax" at the beginning
                pt_model_class = getattr(transformers, pt_model_class_name)

                batch_size, seq_length = pt_inputs["input_ids"].shape
                rnd_start_indices = np.random.randint(0, seq_length - 1, size=(batch_size,))
                for batch_idx, start_index in enumerate(rnd_start_indices):
                    pt_inputs["attention_mask"][batch_idx, :start_index] = 0
                    pt_inputs["attention_mask"][batch_idx, start_index:] = 1
                    prepared_inputs_dict["attention_mask"][batch_idx, :start_index] = 0
                    prepared_inputs_dict["attention_mask"][batch_idx, start_index:] = 1
                pt_model = pt_model_class(config).eval()
                fx_model = model_class(config, dtype=jnp.float32)

                fx_state = convert_pytorch_state_dict_to_flax(pt_model.state_dict(), fx_model)
                fx_model.params = fx_state

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs).to_tuple()

                fx_outputs = fx_model(**prepared_inputs_dict).to_tuple()
                self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")
                for fx_output, pt_output in zip(fx_outputs, pt_outputs):
                    self.assert_almost_equals(fx_output[:, -1], pt_output[:, -1].numpy(), 4e-2)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    pt_model.save_pretrained(tmpdirname)
                    fx_model_loaded = model_class.from_pretrained(tmpdirname, from_pt=True)

                fx_outputs_loaded = fx_model_loaded(**prepared_inputs_dict).to_tuple()
                self.assertEqual(
                    len(fx_outputs_loaded), len(pt_outputs), "Output lengths differ between Flax and PyTorch"
                )
                for fx_output_loaded, pt_output in zip(fx_outputs_loaded, pt_outputs):
                    self.assert_almost_equals(fx_output_loaded[:, -1], pt_output[:, -1].numpy(), 4e-2)

    # overwrite from common since `attention_mask` in combination
    # with `causal_mask` behaves slighly differently
    @is_pt_flax_cross_test
    def test_equivalence_flax_to_pt(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                # prepare inputs
                prepared_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                pt_inputs = {k: torch.tensor(v.tolist()) for k, v in prepared_inputs_dict.items()}

                # load corresponding PyTorch class
                pt_model_class_name = model_class.__name__[4:]  # Skip the "Flax" at the beginning
                pt_model_class = getattr(transformers, pt_model_class_name)

                pt_model = pt_model_class(config).eval()
                fx_model = model_class(config, dtype=jnp.float32)

                pt_model = load_flax_weights_in_pytorch_model(pt_model, fx_model.params)
                batch_size, seq_length = pt_inputs["input_ids"].shape
                rnd_start_indices = np.random.randint(0, seq_length - 1, size=(batch_size,))
                for batch_idx, start_index in enumerate(rnd_start_indices):
                    pt_inputs["attention_mask"][batch_idx, :start_index] = 0
                    pt_inputs["attention_mask"][batch_idx, start_index:] = 1
                    prepared_inputs_dict["attention_mask"][batch_idx, :start_index] = 0
                    prepared_inputs_dict["attention_mask"][batch_idx, start_index:] = 1

                # make sure weights are tied in PyTorch
                pt_model.tie_weights()

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs).to_tuple()

                fx_outputs = fx_model(**prepared_inputs_dict).to_tuple()
                self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")
                for fx_output, pt_output in zip(fx_outputs, pt_outputs):
                    self.assert_almost_equals(fx_output[:, -1], pt_output[:, -1].numpy(), 4e-2)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    fx_model.save_pretrained(tmpdirname)
                    pt_model_loaded = pt_model_class.from_pretrained(tmpdirname, from_flax=True)

                with torch.no_grad():
                    pt_outputs_loaded = pt_model_loaded(**pt_inputs).to_tuple()

                self.assertEqual(
                    len(fx_outputs), len(pt_outputs_loaded), "Output lengths differ between Flax and PyTorch"
                )
                for fx_output, pt_output in zip(fx_outputs, pt_outputs_loaded):
                    self.assert_almost_equals(fx_output[:, -1], pt_output[:, -1].numpy(), 4e-2)

    @slow
    def test_model_from_pretrained(self):
        for model_class_name in self.all_model_classes:
            model = model_class_name.from_pretrained("EleutherAI/gpt-neo-125M")
            outputs = model(np.ones((1, 1)))
            self.assertIsNotNone(outputs)
