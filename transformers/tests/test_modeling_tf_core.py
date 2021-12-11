# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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


import copy
import os
import tempfile
from importlib import import_module

from transformers import is_tf_available
from transformers.models.auto import get_values
from transformers.testing_utils import _tf_gpu_memory_limit, require_tf, slow

from .test_modeling_tf_common import ids_tensor


if is_tf_available():
    import numpy as np
    import tensorflow as tf

    from transformers import (
        TF_MODEL_FOR_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_MASKED_LM_MAPPING,
        TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
        TF_MODEL_FOR_PRETRAINING_MAPPING,
        TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        TFSharedEmbeddings,
    )

    if _tf_gpu_memory_limit is not None:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            # Restrict TensorFlow to only allocate x GB of memory on the GPUs
            try:
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=_tf_gpu_memory_limit)]
                )
                logical_gpus = tf.config.list_logical_devices("GPU")
                print("Logical GPUs", logical_gpus)
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)


@require_tf
class TFCoreModelTesterMixin:

    model_tester = None
    all_model_classes = ()
    all_generative_model_classes = ()
    test_mismatched_shapes = True
    test_resize_embeddings = True
    test_head_masking = True
    is_encoder_decoder = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False) -> dict:
        inputs_dict = copy.deepcopy(inputs_dict)

        if model_class in get_values(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
            inputs_dict = {
                k: tf.tile(tf.expand_dims(v, 1), (1, self.model_tester.num_choices) + (1,) * (v.ndim - 1))
                if isinstance(v, tf.Tensor) and v.ndim > 0
                else v
                for k, v in inputs_dict.items()
            }

        if return_labels:
            if model_class in get_values(TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING):
                inputs_dict["labels"] = tf.ones(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in get_values(TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING):
                inputs_dict["start_positions"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
                inputs_dict["end_positions"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in [
                *get_values(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                *get_values(TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in get_values(TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING):
                inputs_dict["next_sentence_label"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int32)
            elif model_class in [
                *get_values(TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(TF_MODEL_FOR_CAUSAL_LM_MAPPING),
                *get_values(TF_MODEL_FOR_MASKED_LM_MAPPING),
                *get_values(TF_MODEL_FOR_PRETRAINING_MAPPING),
                *get_values(TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING),
            ]:
                inputs_dict["labels"] = tf.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=tf.int32
                )
        return inputs_dict

    @slow
    def test_graph_mode(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config)

            @tf.function
            def run_in_graph_mode():
                return model(inputs)

            outputs = run_in_graph_mode()
            self.assertIsNotNone(outputs)

    @slow
    def test_xla_mode(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            inputs = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config)

            @tf.function(experimental_compile=True)
            def run_in_graph_mode():
                return model(inputs)

            outputs = run_in_graph_mode()
            self.assertIsNotNone(outputs)

    @slow
    def test_saved_model_creation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = False
        config.output_attentions = False

        if hasattr(config, "use_cache"):
            config.use_cache = False

        model_class = self.all_model_classes[0]

        class_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
        model = model_class(config)

        model(class_inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, saved_model=True)
            saved_model_dir = os.path.join(tmpdirname, "saved_model", "1")
            self.assertTrue(os.path.exists(saved_model_dir))

    @slow
    def test_saved_model_creation_extended(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        if hasattr(config, "use_cache"):
            config.use_cache = True

        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", self.model_tester.seq_length)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        for model_class in self.all_model_classes:
            class_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config)
            num_out = len(model(class_inputs_dict))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=True)
                saved_model_dir = os.path.join(tmpdirname, "saved_model", "1")
                model = tf.keras.models.load_model(saved_model_dir)
                outputs = model(class_inputs_dict)

                if self.is_encoder_decoder:
                    output_hidden_states = outputs["encoder_hidden_states"]
                    output_attentions = outputs["encoder_attentions"]
                else:
                    output_hidden_states = outputs["hidden_states"]
                    output_attentions = outputs["attentions"]

                self.assertEqual(len(outputs), num_out)

                expected_num_layers = getattr(
                    self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
                )

                self.assertEqual(len(output_hidden_states), expected_num_layers)
                self.assertListEqual(
                    list(output_hidden_states[0].shape[-2:]),
                    [self.model_tester.seq_length, self.model_tester.hidden_size],
                )

                self.assertEqual(len(output_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(output_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

    @slow
    def test_mixed_precision(self):
        tf.keras.mixed_precision.experimental.set_policy("mixed_float16")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            class_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            model = model_class(config)
            outputs = model(class_inputs_dict)

            self.assertIsNotNone(outputs)

        tf.keras.mixed_precision.experimental.set_policy("float32")

    @slow
    def test_train_pipeline_custom_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        # head_mask and decoder_head_mask has different shapes than other input args
        if "head_mask" in inputs_dict:
            del inputs_dict["head_mask"]
        if "decoder_head_mask" in inputs_dict:
            del inputs_dict["decoder_head_mask"]
        if "cross_attn_head_mask" in inputs_dict:
            del inputs_dict["cross_attn_head_mask"]
        tf_main_layer_classes = set(
            module_member
            for model_class in self.all_model_classes
            for module in (import_module(model_class.__module__),)
            for module_member_name in dir(module)
            if module_member_name.endswith("MainLayer")
            for module_member in (getattr(module, module_member_name),)
            if isinstance(module_member, type)
            and tf.keras.layers.Layer in module_member.__bases__
            and getattr(module_member, "_keras_serializable", False)
        )

        for main_layer_class in tf_main_layer_classes:
            # T5MainLayer needs an embed_tokens parameter when called without the inputs_embeds parameter
            if "T5" in main_layer_class.__name__:
                # Take the same values than in TFT5ModelTester for this shared layer
                shared = TFSharedEmbeddings(self.model_tester.vocab_size, self.model_tester.hidden_size, name="shared")
                config.use_cache = False
                main_layer = main_layer_class(config, embed_tokens=shared)
            else:
                main_layer = main_layer_class(config)

            symbolic_inputs = {
                name: tf.keras.Input(tensor.shape[1:], dtype=tensor.dtype) for name, tensor in inputs_dict.items()
            }

            if hasattr(self.model_tester, "num_labels"):
                num_labels = self.model_tester.num_labels
            else:
                num_labels = 2

            X = tf.data.Dataset.from_tensor_slices(
                (inputs_dict, np.ones((self.model_tester.batch_size, self.model_tester.seq_length, num_labels, 1)))
            ).batch(1)

            hidden_states = main_layer(symbolic_inputs)[0]
            outputs = tf.keras.layers.Dense(num_labels, activation="softmax", name="outputs")(hidden_states)
            model = tf.keras.models.Model(inputs=symbolic_inputs, outputs=[outputs])

            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
            model.fit(X, epochs=1)

            with tempfile.TemporaryDirectory() as tmpdirname:
                filepath = os.path.join(tmpdirname, "keras_model.h5")
                model.save(filepath)
                if "T5" in main_layer_class.__name__:
                    model = tf.keras.models.load_model(
                        filepath,
                        custom_objects={
                            main_layer_class.__name__: main_layer_class,
                            "TFSharedEmbeddings": TFSharedEmbeddings,
                        },
                    )
                else:
                    model = tf.keras.models.load_model(
                        filepath, custom_objects={main_layer_class.__name__: main_layer_class}
                    )
                assert isinstance(model, tf.keras.Model)
                model(inputs_dict)

    @slow
    def test_graph_mode_with_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)

            inputs = copy.deepcopy(inputs_dict)

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
            else:
                inputs["inputs_embeds"] = model.get_input_embeddings()(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = model.get_input_embeddings()(decoder_input_ids)

            inputs = self._prepare_for_class(inputs, model_class)

            @tf.function
            def run_in_graph_mode():
                return model(inputs)

            outputs = run_in_graph_mode()
            self.assertIsNotNone(outputs)

    def _generate_random_bad_tokens(self, num_bad_tokens, model):
        # special tokens cannot be bad tokens
        special_tokens = []
        if model.config.bos_token_id is not None:
            special_tokens.append(model.config.bos_token_id)
        if model.config.pad_token_id is not None:
            special_tokens.append(model.config.pad_token_id)
        if model.config.eos_token_id is not None:
            special_tokens.append(model.config.eos_token_id)

        # create random bad tokens that are not special tokens
        bad_tokens = []
        while len(bad_tokens) < num_bad_tokens:
            token = tf.squeeze(ids_tensor((1, 1), self.model_tester.vocab_size), 0).numpy()[0]
            if token not in special_tokens:
                bad_tokens.append(token)
        return bad_tokens

    def _check_generated_ids(self, output_ids):
        for token_id in output_ids[0].numpy().tolist():
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, self.model_tester.vocab_size)

    def _check_match_tokens(self, generated_ids, bad_words_ids):
        # for all bad word tokens
        for bad_word_ids in bad_words_ids:
            # for all slices in batch
            for generated_ids_slice in generated_ids:
                # for all word idx
                for i in range(len(bad_word_ids), len(generated_ids_slice)):
                    # if tokens match
                    if generated_ids_slice[i - len(bad_word_ids) : i] == bad_word_ids:
                        return True
        return False
