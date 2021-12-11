# coding=utf-8
# Copyright 2021 HuggingFace Inc.
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


import itertools
import random
import unittest

import numpy as np

from transformers import is_speech_available
from transformers.testing_utils import require_torch, require_torchaudio

from .test_sequence_feature_extraction_common import SequenceFeatureExtractionTestMixin


if is_speech_available():
    from transformers import Speech2TextFeatureExtractor

global_rng = random.Random()


def floats_list(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    values = []
    for batch_idx in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)

    return values


@require_torch
@require_torchaudio
class Speech2TextFeatureExtractionTester(unittest.TestCase):
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=24,
        num_mel_bins=24,
        padding_value=0.0,
        sampling_rate=16_000,
        return_attention_mask=True,
        do_normalize=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.num_mel_bins = num_mel_bins
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "num_mel_bins": self.num_mel_bins,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "return_attention_mask": self.return_attention_mask,
            "do_normalize": self.do_normalize,
        }

    def prepare_inputs_for_common(self, equal_length=False, numpify=False):
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_length:
            speech_inputs = [floats_list((self.max_seq_length, self.feature_size)) for _ in range(self.batch_size)]
        else:
            # make sure that inputs increase in size
            speech_inputs = [
                floats_list((x, self.feature_size))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]
        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]
        return speech_inputs


@require_torch
@require_torchaudio
class Speech2TextFeatureExtractionTest(SequenceFeatureExtractionTestMixin, unittest.TestCase):

    feature_extraction_class = Speech2TextFeatureExtractor if is_speech_available() else None

    def setUp(self):
        self.feat_extract_tester = Speech2TextFeatureExtractionTester(self)

    def _check_zero_mean_unit_variance(self, input_vector):
        self.assertTrue(np.all(np.mean(input_vector, axis=0) < 1e-3))
        self.assertTrue(np.all(np.abs(np.var(input_vector, axis=0) - 1) < 1e-3))

    def test_call(self):
        # Tests that all call wrap to encode_plus and batch_encode_plus
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        # create three inputs of length 800, 1000, and 1200
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        np_speech_inputs = [np.asarray(speech_input) for speech_input in speech_inputs]

        # Test feature size
        input_features = feature_extractor(np_speech_inputs, padding=True, return_tensors="np").input_features
        self.assertTrue(input_features.ndim == 3)
        self.assertTrue(input_features.shape[-1] == feature_extractor.feature_size)

        # Test not batched input
        encoded_sequences_1 = feature_extractor(speech_inputs[0], return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs[0], return_tensors="np").input_features
        self.assertTrue(np.allclose(encoded_sequences_1, encoded_sequences_2, atol=1e-3))

        # Test batched
        encoded_sequences_1 = feature_extractor(speech_inputs, return_tensors="np").input_features
        encoded_sequences_2 = feature_extractor(np_speech_inputs, return_tensors="np").input_features
        for enc_seq_1, enc_seq_2 in zip(encoded_sequences_1, encoded_sequences_2):
            self.assertTrue(np.allclose(enc_seq_1, enc_seq_2, atol=1e-3))

    def test_cepstral_mean_and_variance_normalization(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]

        paddings = ["longest", "max_length", "do_not_pad"]
        max_lengths = [None, 16, None]
        for max_length, padding in zip(max_lengths, paddings):
            inputs = feature_extractor(
                speech_inputs, padding=padding, max_length=max_length, return_attention_mask=True
            )
            input_features = inputs.input_features
            attention_mask = inputs.attention_mask
            fbank_feat_lengths = [np.sum(x) for x in attention_mask]

            self._check_zero_mean_unit_variance(input_features[0][: fbank_feat_lengths[0]])
            self._check_zero_mean_unit_variance(input_features[1][: fbank_feat_lengths[1]])
            self._check_zero_mean_unit_variance(input_features[2][: fbank_feat_lengths[2]])

    def test_cepstral_mean_and_variance_normalization_np(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]

        paddings = ["longest", "max_length", "do_not_pad"]
        max_lengths = [None, 16, None]
        for max_length, padding in zip(max_lengths, paddings):
            inputs = feature_extractor(
                speech_inputs, max_length=max_length, padding=padding, return_tensors="np", return_attention_mask=True
            )
            input_features = inputs.input_features
            attention_mask = inputs.attention_mask
            fbank_feat_lengths = [np.sum(x) for x in attention_mask]

            self._check_zero_mean_unit_variance(input_features[0][: fbank_feat_lengths[0]])
            self.assertTrue(input_features[0][fbank_feat_lengths[0] :].sum() < 1e-6)
            self._check_zero_mean_unit_variance(input_features[1][: fbank_feat_lengths[1]])
            self.assertTrue(input_features[0][fbank_feat_lengths[1] :].sum() < 1e-6)
            self._check_zero_mean_unit_variance(input_features[2][: fbank_feat_lengths[2]])

    def test_cepstral_mean_and_variance_normalization_trunc_max_length(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        inputs = feature_extractor(
            speech_inputs,
            padding="max_length",
            max_length=4,
            truncation=True,
            return_tensors="np",
            return_attention_mask=True,
        )
        input_features = inputs.input_features
        attention_mask = inputs.attention_mask
        fbank_feat_lengths = np.sum(attention_mask == 1, axis=1)

        self._check_zero_mean_unit_variance(input_features[0, : fbank_feat_lengths[0]])
        self._check_zero_mean_unit_variance(input_features[1])
        self._check_zero_mean_unit_variance(input_features[2])

    def test_cepstral_mean_and_variance_normalization_trunc_longest(self):
        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        inputs = feature_extractor(
            speech_inputs,
            padding="longest",
            max_length=4,
            truncation=True,
            return_tensors="np",
            return_attention_mask=True,
        )
        input_features = inputs.input_features
        attention_mask = inputs.attention_mask
        fbank_feat_lengths = np.sum(attention_mask == 1, axis=1)

        self._check_zero_mean_unit_variance(input_features[0, : fbank_feat_lengths[0]])
        self._check_zero_mean_unit_variance(input_features[1, : fbank_feat_lengths[1]])
        self._check_zero_mean_unit_variance(input_features[2])

        # make sure that if max_length < longest -> then pad to max_length
        self.assertEqual(input_features.shape, (3, 4, 24))

        speech_inputs = [floats_list((1, x))[0] for x in range(800, 1400, 200)]
        inputs = feature_extractor(
            speech_inputs,
            padding="longest",
            max_length=16,
            truncation=True,
            return_tensors="np",
            return_attention_mask=True,
        )
        input_features = inputs.input_features
        attention_mask = inputs.attention_mask
        fbank_feat_lengths = np.sum(attention_mask == 1, axis=1)

        self._check_zero_mean_unit_variance(input_features[0, : fbank_feat_lengths[0]])
        self._check_zero_mean_unit_variance(input_features[1, : fbank_feat_lengths[1]])
        self._check_zero_mean_unit_variance(input_features[2])

        # make sure that if max_length < longest -> then pad to max_length
        self.assertEqual(input_features.shape, (3, 6, 24))

    def test_double_precision_pad(self):
        import torch

        feature_extractor = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        np_speech_inputs = np.random.rand(100, 32).astype(np.float64)
        py_speech_inputs = np_speech_inputs.tolist()

        for inputs in [py_speech_inputs, np_speech_inputs]:
            np_processed = feature_extractor.pad([{"input_features": inputs}], return_tensors="np")
            self.assertTrue(np_processed.input_features.dtype == np.float32)
            pt_processed = feature_extractor.pad([{"input_features": inputs}], return_tensors="pt")
            self.assertTrue(pt_processed.input_features.dtype == torch.float32)
