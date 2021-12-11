# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import math
import os
import re
import sys
import unittest
from unittest.mock import patch

from parameterized import parameterized
from transformers.file_utils import is_apex_available
from transformers.integrations import is_fairscale_available
from transformers.testing_utils import (
    CaptureStderr,
    ExtendSysPath,
    TestCasePlus,
    execute_subprocess_async,
    get_gpu_count,
    get_torch_dist_unique_port,
    require_torch,
    require_torch_gpu,
    require_torch_multi_gpu,
    require_torch_non_multi_gpu,
    slow,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import set_seed


bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/../../examples/pytorch/translation"):
    from run_translation import main  # noqa


set_seed(42)
MARIAN_MODEL = "sshleifer/student_marian_en_ro_6_1"
MBART_TINY = "sshleifer/tiny-mbart"


# a candidate for testing_utils
def require_fairscale(test_case):
    """
    Decorator marking a test that requires fairscale
    """
    if not is_fairscale_available():
        return unittest.skip("test requires fairscale")(test_case)
    else:
        return test_case


# a candidate for testing_utils
def require_apex(test_case):
    """
    Decorator marking a test that requires apex
    """
    if not is_apex_available():
        return unittest.skip("test requires apex")(test_case)
    else:
        return test_case


@require_torch
class TestTrainerExt(TestCasePlus):
    def run_seq2seq_quick(
        self,
        distributed=False,
        extra_args_str=None,
        predict_with_generate=True,
        do_train=True,
        do_eval=True,
        do_predict=True,
    ):
        output_dir = self.run_trainer(
            eval_steps=1,
            max_len=12,
            model_name=MBART_TINY,
            num_train_epochs=1,
            distributed=distributed,
            extra_args_str=extra_args_str,
            predict_with_generate=predict_with_generate,
            do_train=do_train,
            do_eval=do_eval,
            do_predict=do_predict,
        )
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history

        if not do_eval:
            return

        eval_metrics = [log for log in logs if "eval_loss" in log.keys()]

        first_step_stats = eval_metrics[0]
        if predict_with_generate:
            assert "eval_bleu" in first_step_stats

            last_step_stats = eval_metrics[-1]
            assert isinstance(last_step_stats["eval_bleu"], float)
            assert not math.isnan(float(last_step_stats["eval_loss"])), "eval_loss must not be `nan`"

    @require_torch_non_multi_gpu
    def test_run_seq2seq_no_dist(self):
        self.run_seq2seq_quick()

    # verify that the trainer can handle non-distributed with n_gpu > 1
    @require_torch_multi_gpu
    def test_run_seq2seq_dp(self):
        self.run_seq2seq_quick(distributed=False)

    # verify that the trainer can handle distributed with n_gpu > 1
    @require_torch_multi_gpu
    def test_run_seq2seq_ddp(self):
        self.run_seq2seq_quick(distributed=True)

    # test --sharded_ddp w/o --fp16
    @require_torch_multi_gpu
    @require_fairscale
    def test_run_seq2seq_sharded_ddp(self):
        self.run_seq2seq_quick(distributed=True, extra_args_str="--sharded_ddp simple")

    # test --sharded_ddp w/ --fp16
    @require_torch_multi_gpu
    @require_fairscale
    def test_run_seq2seq_sharded_ddp_fp16(self):
        self.run_seq2seq_quick(distributed=True, extra_args_str="--sharded_ddp simple --fp16")

    # test --sharded_ddp zero_dp_2 w/o --fp16
    @require_torch_multi_gpu
    @require_fairscale
    def test_run_seq2seq_fully_sharded_ddp(self):
        self.run_seq2seq_quick(distributed=True, extra_args_str="--sharded_ddp zero_dp_2", predict_with_generate=False)

    # test --sharded_ddp zero_dp_2 w/ --fp16
    @require_torch_multi_gpu
    @require_fairscale
    def test_run_seq2seq_fully_sharded_ddp_fp16(self):
        self.run_seq2seq_quick(
            distributed=True, extra_args_str="--sharded_ddp zero_dp_2 --fp16", predict_with_generate=False
        )

    @require_apex
    @require_torch_gpu
    def test_run_seq2seq_apex(self):
        # XXX: apex breaks the trainer if it's run twice e.g. run_seq2seq.main() from the same
        # program and it breaks other tests that run from the same pytest worker, therefore until this is
        # sorted out it must be run only in an external program, that is distributed=True in this
        # test and only under one or more gpus - if we want cpu will need to make a special test
        #
        # specifically to the problem traced it to self.optimizer.step() - if it's run 2nd time via
        # 2nd main() call it botches the future eval.
        #
        self.run_seq2seq_quick(distributed=True, extra_args_str="--fp16 --fp16_backend=apex")
        # test 2nd time - was getting eval_loss': nan'
        # to reproduce the problem set distributed=False
        self.run_seq2seq_quick(distributed=True, extra_args_str="--fp16 --fp16_backend=apex")

    @parameterized.expand(["base", "low", "high", "mixed"])
    @require_torch_multi_gpu
    def test_trainer_log_level_replica(self, experiment_id):
        # as each sub-test is slow-ish split into multiple sub-tests to avoid CI timeout
        experiments = dict(
            # test with the default log_level - should be info and thus log info once
            base=dict(extra_args_str="", n_matches=1),
            # test with low log_level and log_level_replica - should be noisy on all processes
            # now the info string should appear twice on 2 processes
            low=dict(extra_args_str="--log_level debug --log_level_replica debug", n_matches=2),
            # test with high log_level and low log_level_replica
            # now the info string should appear once only on the replica
            high=dict(extra_args_str="--log_level error --log_level_replica debug", n_matches=1),
            # test with high log_level and log_level_replica - should be quiet on all processes
            mixed=dict(extra_args_str="--log_level error --log_level_replica error", n_matches=0),
        )

        data = experiments[experiment_id]
        kwargs = dict(distributed=True, predict_with_generate=False, do_eval=False, do_predict=False)
        log_info_string = "Running training"
        with CaptureStderr() as cl:
            self.run_seq2seq_quick(**kwargs, extra_args_str=data["extra_args_str"])
        n_matches = len(re.findall(log_info_string, cl.err))
        self.assertEqual(n_matches, data["n_matches"])

    @slow
    def test_run_seq2seq_slow(self):
        output_dir = self.run_trainer(
            eval_steps=2,
            max_len=128,
            model_name=MARIAN_MODEL,
            learning_rate=3e-4,
            num_train_epochs=10,
            distributed=False,
        )

        # Check metrics
        logs = TrainerState.load_from_json(os.path.join(output_dir, "trainer_state.json")).log_history
        eval_metrics = [log for log in logs if "eval_loss" in log.keys()]
        first_step_stats = eval_metrics[0]
        last_step_stats = eval_metrics[-1]

        assert first_step_stats["eval_loss"] > last_step_stats["eval_loss"], "model learned nothing"
        assert isinstance(last_step_stats["eval_bleu"], float)

        # test if do_predict saves generations and metrics
        contents = os.listdir(output_dir)
        contents = {os.path.basename(p) for p in contents}
        assert "generated_predictions.txt" in contents
        assert "predict_results.json" in contents

    def run_trainer(
        self,
        eval_steps: int,
        max_len: int,
        model_name: str,
        num_train_epochs: int,
        learning_rate: float = 3e-3,
        distributed: bool = False,
        extra_args_str: str = None,
        predict_with_generate: bool = True,
        do_train: bool = True,
        do_eval: bool = True,
        do_predict: bool = True,
    ):
        data_dir = self.test_file_dir / "../fixtures/tests_samples/wmt_en_ro"
        output_dir = self.get_auto_remove_tmp_dir()
        args_train = f"""
            --model_name_or_path {model_name}
            --train_file {data_dir}/train.json
            --validation_file {data_dir}/val.json
            --test_file {data_dir}/test.json
            --output_dir {output_dir}
            --overwrite_output_dir
            --max_train_samples 8
            --max_source_length {max_len}
            --max_target_length {max_len}
            --do_train
            --num_train_epochs {str(num_train_epochs)}
            --per_device_train_batch_size 4
            --learning_rate {learning_rate}
            --warmup_steps 8
            --logging_steps 0
            --logging_strategy no
            --save_steps {str(eval_steps)}
            --group_by_length
            --label_smoothing_factor 0.1
            --adafactor
            --target_lang ro_RO
            --source_lang en_XX
        """

        args_eval = f"""
            --do_eval
            --per_device_eval_batch_size 4
            --max_eval_samples 8
            --val_max_target_length {max_len}
            --evaluation_strategy steps
            --eval_steps {str(eval_steps)}
        """

        args_predict = """
            --do_predict
        """

        args = ""
        if do_train:
            args += args_train

        if do_eval:
            args += args_eval

        if do_predict:
            args += args_predict

        if predict_with_generate:
            args += "--predict_with_generate"

        args = args.split()

        if extra_args_str is not None:
            args.extend(extra_args_str.split())

        if distributed:
            n_gpu = get_gpu_count()
            master_port = get_torch_dist_unique_port()
            distributed_args = f"""
                -m torch.distributed.launch
                --nproc_per_node={n_gpu}
                --master_port={master_port}
                {self.examples_dir_str}/pytorch/translation/run_translation.py
            """.split()
            cmd = [sys.executable] + distributed_args + args
            execute_subprocess_async(cmd, env=self.get_env())
        else:
            testargs = ["run_translation.py"] + args
            with patch.object(sys, "argv", testargs):
                main()

        return output_dir
