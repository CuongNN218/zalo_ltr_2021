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

import concurrent.futures
import json
import os
import shutil
import tempfile
import unittest

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.testing_utils import require_tokenizers

from .test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class PreTrainedTokenizationFastTest(TokenizerTesterMixin, unittest.TestCase):
    rust_tokenizer_class = PreTrainedTokenizerFast
    test_slow_tokenizer = False
    test_rust_tokenizer = True
    from_pretrained_vocab_key = "tokenizer_file"

    def setUp(self):
        self.test_rust_tokenizer = False  # because we don't have pretrained_vocab_files_map
        super().setUp()
        self.test_rust_tokenizer = True

        model_paths = ["robot-test/dummy-tokenizer-fast", "robot-test/dummy-tokenizer-wordlevel"]

        # Inclusion of 2 tokenizers to test different types of models (Unigram and WordLevel for the moment)
        self.tokenizers_list = [(PreTrainedTokenizerFast, model_path, {}) for model_path in model_paths]

        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_paths[0])
        tokenizer.save_pretrained(self.tmpdirname)

    def test_tokenizer_mismatch_warning(self):
        # We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any
        # model
        pass

    def test_pretrained_model_lists(self):
        # We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any
        # model
        pass

    def test_prepare_for_model(self):
        # We disable this test for PreTrainedTokenizerFast because it is the only tokenizer that is not linked to any
        # model
        pass

    def test_rust_tokenizer_signature(self):
        # PreTrainedTokenizerFast doesn't have tokenizer_file in its signature
        pass

    def test_training_new_tokenizer(self):
        tmpdirname_orig = self.tmpdirname
        # Here we want to test the 2 available tokenizers that use 2 different types of models: Unigram and WordLevel.
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                try:
                    self.tmpdirname = tempfile.mkdtemp()
                    tokenizer = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                    tokenizer.save_pretrained(self.tmpdirname)
                    super().test_training_new_tokenizer()
                finally:
                    # Even if the test fails, we must be sure that the folder is deleted and that the default tokenizer
                    # is restored
                    shutil.rmtree(self.tmpdirname)
                    self.tmpdirname = tmpdirname_orig

    def test_training_new_tokenizer_with_special_tokens_change(self):
        tmpdirname_orig = self.tmpdirname
        # Here we want to test the 2 available tokenizers that use 2 different types of models: Unigram and WordLevel.
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                try:
                    self.tmpdirname = tempfile.mkdtemp()
                    tokenizer = self.rust_tokenizer_class.from_pretrained(pretrained_name, **kwargs)

                    tokenizer.save_pretrained(self.tmpdirname)
                    super().test_training_new_tokenizer_with_special_tokens_change()
                finally:
                    # Even if the test fails, we must be sure that the folder is deleted and that the default tokenizer
                    # is restored
                    shutil.rmtree(self.tmpdirname)
                    self.tmpdirname = tmpdirname_orig


@require_tokenizers
class TokenizerVersioningTest(unittest.TestCase):
    def test_local_versioning(self):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        json_tokenizer = json.loads(tokenizer._tokenizer.to_str())
        json_tokenizer["model"]["vocab"]["huggingface"] = len(tokenizer)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            json.dump(json_tokenizer, open(os.path.join(tmp_dir, "tokenizer.4.0.0.json"), "w"))

            # This should pick the new tokenizer file as the version of Transformers is > 4.0.0
            new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            self.assertEqual(len(new_tokenizer), len(tokenizer) + 1)
            json_tokenizer = json.loads(new_tokenizer._tokenizer.to_str())
            self.assertIn("huggingface", json_tokenizer["model"]["vocab"])

            # Will need to be adjusted if we reach v42 and this test is still here.
            # Should pick the old tokenizer file as the version of Transformers is < 4.0.0
            shutil.move(os.path.join(tmp_dir, "tokenizer.4.0.0.json"), os.path.join(tmp_dir, "tokenizer.42.0.0.json"))
            new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)
            self.assertEqual(len(new_tokenizer), len(tokenizer))
            json_tokenizer = json.loads(new_tokenizer._tokenizer.to_str())
            self.assertNotIn("huggingface", json_tokenizer["model"]["vocab"])

    def test_repo_versioning(self):
        # This repo has two tokenizer files, one for v4.0.0 and above with an added token, one for versions lower.
        repo = "sgugger/finetuned-bert-mrpc"

        # This should pick the new tokenizer file as the version of Transformers is > 4.0.0
        tokenizer = AutoTokenizer.from_pretrained(repo)
        self.assertEqual(len(tokenizer), 28997)
        json_tokenizer = json.loads(tokenizer._tokenizer.to_str())
        self.assertIn("huggingface", json_tokenizer["model"]["vocab"])

        # Testing an older version by monkey-patching the version in the module it's used.
        import transformers as old_transformers

        old_transformers.tokenization_utils_base.__version__ = "3.0.0"
        old_tokenizer = old_transformers.models.auto.AutoTokenizer.from_pretrained(repo)
        self.assertEqual(len(old_tokenizer), 28996)
        json_tokenizer = json.loads(old_tokenizer._tokenizer.to_str())
        self.assertNotIn("huggingface", json_tokenizer["model"]["vocab"])


@require_tokenizers
class ReduceMutableBorrowTests(unittest.TestCase):
    def test_async_share_tokenizer(self):
        # See https://github.com/huggingface/transformers/pull/12550
        # and https://github.com/huggingface/tokenizers/issues/537
        tokenizer = PreTrainedTokenizerFast.from_pretrained("robot-test/dummy-tokenizer-wordlevel")
        text = "The Matrix is a 1999 science fiction action film."

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.fetch, tokenizer, text) for i in range(10)]
            return_value = [future.result() for future in futures]
            self.assertEqual(return_value, [[1, 10, 0, 8, 0, 18, 0, 0, 0, 2] for i in range(10)])

    def fetch(self, tokenizer, text):
        return tokenizer.encode(text, truncation="longest_first", padding="longest")
