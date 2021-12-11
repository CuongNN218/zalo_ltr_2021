..
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.


DeepSpeed Integration
-----------------------------------------------------------------------------------------------------------------------


`DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ implements everything described in the `ZeRO paper
<https://arxiv.org/abs/1910.02054>`__. Currently it provides full support for:

1. Optimizer state partitioning (ZeRO stage 1)
2. Gradient partitioning (ZeRO stage 2)
3. Parameter partitioning (ZeRO stage 3)
4. Custom mixed precision training handling
5. A range of fast CUDA-extension-based optimizers
6. ZeRO-Offload to CPU and NVMe

ZeRO-Offload has its own dedicated paper: `ZeRO-Offload: Democratizing Billion-Scale Model Training
<https://arxiv.org/abs/2101.06840>`__. And NVMe-support is described in the paper `ZeRO-Infinity: Breaking the GPU
Memory Wall for Extreme Scale Deep Learning <https://arxiv.org/abs/2104.07857>`__.

DeepSpeed ZeRO-2 is primarily used only for training, as its features are of no use to inference.

DeepSpeed ZeRO-3 can be used for inference as well, since it allows huge models to be loaded on multiple GPUs, which
won't be possible on a single GPU.



🤗 Transformers integrates `DeepSpeed <https://github.com/microsoft/DeepSpeed>`__ via 2 options:

1. Integration of the core DeepSpeed features via :class:`~transformers.Trainer`. This is everything done for you type
   of integration - just supply your custom config file or use our template and you have nothing else to do. Most of
   this document is focused on this feature.
2. If you don't use :class:`~transformers.Trainer` and want to use your own Trainer where you integrated DeepSpeed
   yourself, core functionality functions like ``from_pretrained`` and ``from_config`` include integration of essential
   parts of DeepSpeed like ``zero.Init`` for ZeRO stage 3 and higher. To tap into this feature read the docs on
   :ref:`deepspeed-non-trainer-integration`.

What is integrated:

Training:

1. DeepSpeed ZeRO training supports the full ZeRO stages 1, 2 and 3 with ZeRO-Infinity (CPU and NVME offload).

Inference:

1. DeepSpeed ZeRO Inference supports ZeRO stage 3 with ZeRO-Infinity. It uses the same ZeRO protocol as training, but
   it doesn't use an optimizer and a lr scheduler and only stage 3 is relevant. For more details see:
   :ref:`deepspeed-zero-inference`.

There is also DeepSpeed Inference - this is a totally different technology which uses Tensor Parallelism instead of
ZeRO (coming soon).



.. _deepspeed-trainer-integration:


Trainer Deepspeed Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. _deepspeed-installation:

Installation
=======================================================================================================================

Install the library via pypi:

.. code-block:: bash

    pip install deepspeed

or via ``transformers``' ``extras``:

.. code-block:: bash

    pip install transformers[deepspeed]

or find more details on `the DeepSpeed's GitHub page <https://github.com/microsoft/deepspeed#installation>`__ and
`advanced install <https://www.deepspeed.ai/tutorials/advanced-install/>`__.

If you're still struggling with the build, first make sure to read :ref:`zero-install-notes`.

If you don't prebuild the extensions and rely on them to be built at run time and you tried all of the above solutions
to no avail, the next thing to try is to pre-build the modules before installing them.

To make a local build for DeepSpeed:

.. code-block:: bash

    git clone https://github.com/microsoft/DeepSpeed/
    cd DeepSpeed
    rm -rf build
    TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
    --global-option="build_ext" --global-option="-j8" --no-cache -v \
    --disable-pip-version-check 2>&1 | tee build.log

If you intend to use NVMe offload you will need to also include ``DS_BUILD_AIO=1`` in the instructions above (and also
install `libaio-dev` system-wide).

Edit ``TORCH_CUDA_ARCH_LIST`` to insert the code for the architectures of the GPU cards you intend to use. Assuming all
your cards are the same you can get the arch via:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.get_device_capability())"

So if you get ``8, 6``, then use ``TORCH_CUDA_ARCH_LIST="8.6"``. If you have multiple different cards, you can list all
of them like so ``TORCH_CUDA_ARCH_LIST="6.1;8.6"``

If you need to use the same setup on multiple machines, make a binary wheel:

.. code-block:: bash

    git clone https://github.com/microsoft/DeepSpeed/
    cd DeepSpeed
    rm -rf build
    TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 \
    python setup.py build_ext -j8 bdist_wheel

it will generate something like ``dist/deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`` which now you can install
as ``pip install deepspeed-0.3.13+8cd046f-cp38-cp38-linux_x86_64.whl`` locally or on any other machine.

Again, remember to ensure to adjust ``TORCH_CUDA_ARCH_LIST`` to the target architectures.

You can find the complete list of NVIDIA GPUs and their corresponding **Compute Capabilities** (same as arch in this
context) `here <https://developer.nvidia.com/cuda-gpus>`__.

You can check the archs pytorch was built with using:

.. code-block:: bash

    python -c "import torch; print(torch.cuda.get_arch_list())"

Here is how to find out the arch for one of the installed GPU. For example, for GPU 0:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 python -c "import torch; \
    print(torch.cuda.get_device_properties(torch.device('cuda')))"

If the output is:

.. code-block:: bash

    _CudaDeviceProperties(name='GeForce RTX 3090', major=8, minor=6, total_memory=24268MB, multi_processor_count=82)

then you know that this card's arch is ``8.6``.

You can also leave ``TORCH_CUDA_ARCH_LIST`` out completely and then the build program will automatically query the
architecture of the GPUs the build is made on. This may or may not match the GPUs on the target machines, that's why
it's best to specify the desired archs explicitly.

If after trying everything suggested you still encounter build issues, please, proceed with the GitHub Issue of
`Deepspeed <https://github.com/microsoft/DeepSpeed/issues>`__,



.. _deepspeed-multi-gpu:

Deployment with multiple GPUs
=======================================================================================================================

To deploy this feature with multiple GPUs adjust the :class:`~transformers.Trainer` command line arguments as
following:

1. replace ``python -m torch.distributed.launch`` with ``deepspeed``.
2. add a new argument ``--deepspeed ds_config.json``, where ``ds_config.json`` is the DeepSpeed configuration file as
   documented `here <https://www.deepspeed.ai/docs/config-json/>`__. The file naming is up to you.

Therefore, if your original command line looked as following:

.. code-block:: bash

    python -m torch.distributed.launch --nproc_per_node=2 your_program.py <normal cl args>

Now it should be:

.. code-block:: bash

    deepspeed --num_gpus=2 your_program.py <normal cl args> --deepspeed ds_config.json

Unlike, ``torch.distributed.launch`` where you have to specify how many GPUs to use with ``--nproc_per_node``, with the
``deepspeed`` launcher you don't have to use the corresponding ``--num_gpus`` if you want all of your GPUs used. The
full details on how to configure various nodes and GPUs can be found `here
<https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node>`__.

In fact, you can continue using ``-m torch.distributed.launch`` with DeepSpeed as long as you don't need to use
``deepspeed`` launcher-specific arguments. Typically if you don't need a multi-node setup you're not required to use
the ``deepspeed`` launcher. But since in the DeepSpeed documentation it'll be used everywhere, for consistency we will
use it here as well.

Here is an example of running ``run_translation.py`` under DeepSpeed deploying all available GPUs:

.. code-block:: bash

    deepspeed examples/pytorch/translation/run_translation.py \
    --deepspeed tests/deepspeed/ds_config_zero3.json \
    --model_name_or_path t5-small --per_device_train_batch_size 1   \
    --output_dir output_dir --overwrite_output_dir --fp16 \
    --do_train --max_train_samples 500 --num_train_epochs 1 \
    --dataset_name wmt16 --dataset_config "ro-en" \
    --source_lang en --target_lang ro


Note that in the DeepSpeed documentation you are likely to see ``--deepspeed --deepspeed_config ds_config.json`` - i.e.
two DeepSpeed-related arguments, but for the sake of simplicity, and since there are already so many arguments to deal
with, we combined the two into a single argument.

For some practical usage examples, please, see this `post
<https://github.com/huggingface/transformers/issues/8771#issuecomment-759248400>`__.



.. _deepspeed-one-gpu:

Deployment with one GPU
=======================================================================================================================

To deploy DeepSpeed with one GPU adjust the :class:`~transformers.Trainer` command line arguments as following:

.. code-block:: bash

    deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
    --deepspeed tests/deepspeed/ds_config_zero2.json \
    --model_name_or_path t5-small --per_device_train_batch_size 1   \
    --output_dir output_dir --overwrite_output_dir --fp16 \
    --do_train --max_train_samples 500 --num_train_epochs 1 \
    --dataset_name wmt16 --dataset_config "ro-en" \
    --source_lang en --target_lang ro

This is almost the same as with multiple-GPUs, but here we tell DeepSpeed explicitly to use just one GPU via
``--num_gpus=1``. By default, DeepSpeed deploys all GPUs it can see on the given node. If you have only 1 GPU to start
with, then you don't need this argument. The following `documentation
<https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node>`__ discusses the launcher options.

Why would you want to use DeepSpeed with just one GPU?

1. It has a ZeRO-offload feature which can delegate some computations and memory to the host's CPU and RAM, and thus
   leave more GPU resources for model's needs - e.g. larger batch size, or enabling a fitting of a very big model which
   normally won't fit.
2. It provides a smart GPU memory management system, that minimizes memory fragmentation, which again allows you to fit
   bigger models and data batches.

While we are going to discuss the configuration in details next, the key to getting a huge improvement on a single GPU
with DeepSpeed is to have at least the following configuration in the configuration file:

.. code-block:: json

    {
      "zero_optimization": {
         "stage": 2,
         "offload_optimizer": {
             "device": "cpu",
             "pin_memory": true
         },
         "allgather_partitions": true,
         "allgather_bucket_size": 2e8,
         "reduce_scatter": true,
         "reduce_bucket_size": 2e8,
         "overlap_comm": true,
         "contiguous_gradients": true
      }
    }

which enables optimizer offload and some other important features. You may experiment with the buffer sizes, you will
find more details in the discussion below.

For a practical usage example of this type of deployment, please, see this `post
<https://github.com/huggingface/transformers/issues/8771#issuecomment-759176685>`__.

You may also try the ZeRO-3 with CPU and NVMe offload as explained further in this document.

<!--- TODO: Benchmark whether we can get better performance out of ZeRO-3 vs. ZeRO-2 on a single GPU, and then
recommend ZeRO-3 config as starting one. -->

Notes:

- if you need to run on a specific GPU, which is different from GPU 0, you can't use ``CUDA_VISIBLE_DEVICES`` to limit
  the visible scope of available GPUs. Instead, you have to use the following syntax:

   .. code-block:: bash

       deepspeed --include localhost:1 examples/pytorch/translation/run_translation.py ...

   In this example, we tell DeepSpeed to use GPU 1 (second gpu).



.. _deepspeed-notebook:

Deployment in Notebooks
=======================================================================================================================

The problem with running notebook cells as a script is that there is no normal ``deepspeed`` launcher to rely on, so
under certain setups we have to emulate it.

If you're using only 1 GPU, here is how you'd have to adjust your training code in the notebook to use DeepSpeed.

.. code-block:: python

    # DeepSpeed requires a distributed environment even when only one process is used.
    # This emulates a launcher in the notebook
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9994' # modify if RuntimeError: Address already in use
    os.environ['RANK'] = "0"
    os.environ['LOCAL_RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"

    # Now proceed as normal, plus pass the deepspeed config file
    training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
    trainer = Trainer(...)
    trainer.train()

Note: ``...`` stands for the normal arguments that you'd pass to the functions.

If you want to use more than 1 GPU, you must use a multi-process environment for DeepSpeed to work. That is, you have
to use the launcher for that purpose and this cannot be accomplished by emulating the distributed environment presented
at the beginning of this section.

If you want to create the config file on the fly in the notebook in the current directory, you could have a dedicated
cell with:

.. code-block:: python

    %%bash
    cat <<'EOT' > ds_config_zero3.json
    {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },

        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": true
            },
            "overlap_comm": true,
            "contiguous_gradients": true,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_fp16_weights_on_model_save": true
        },

        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": false
    }
    EOT


If the training script is in a normal file and not in the notebook cells, you can launch ``deepspeed`` normally via
shell from a cell. For example, to use ``run_translation.py`` you would launch it with:

.. code-block::

    !git clone https://github.com/huggingface/transformers
    !cd transformers; deepspeed examples/pytorch/translation/run_translation.py ...

or with ``%%bash`` magic, where you can write a multi-line code for the shell program to run:

.. code-block::

    %%bash

    git clone https://github.com/huggingface/transformers
    cd transformers
    deepspeed examples/pytorch/translation/run_translation.py ...

In such case you don't need any of the code presented at the beginning of this section.

Note: While ``%%bash`` magic is neat, but currently it buffers the output so you won't see the logs until the process
completes.




.. _deepspeed-config:

Configuration
=======================================================================================================================

For the complete guide to the DeepSpeed configuration options that can be used in its configuration file please refer
to the `following documentation <https://www.deepspeed.ai/docs/config-json/>`__.

You can find dozens of DeepSpeed configuration examples that address various practical needs in `the DeepSpeedExamples
repo <https://github.com/microsoft/DeepSpeedExamples>`__:

.. code-block:: bash

    git clone https://github.com/microsoft/DeepSpeedExamples
    cd DeepSpeedExamples
    find . -name '*json'

Continuing the code from above, let's say you're looking to configure the Lamb optimizer. So you can search through the
example ``.json`` files with:

.. code-block:: bash

    grep -i Lamb $(find . -name '*json')

Some more examples are to be found in the `main repo <https://github.com/microsoft/DeepSpeed>`__ as well.

When using DeepSpeed you always need to supply a DeepSpeed configuration file, yet some configuration parameters have
to be configured via the command line. You will find the nuances in the rest of this guide.

To get an idea of what DeepSpeed configuration file looks like, here is one that activates ZeRO stage 2 features,
including optimizer states cpu offload, uses ``AdamW`` optimizer and ``WarmupLR`` scheduler and will enable mixed
precision training if ``--fp16`` is passed:

.. code-block:: json

    {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },

        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "allgather_partitions": true,
            "allgather_bucket_size": 2e8,
            "overlap_comm": true,
            "reduce_scatter": true,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": true
        },

        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    }

When you execute the program, DeepSpeed will log the configuration it received from the :class:`~transformers.Trainer`
to the console, so you can see exactly what was the final configuration passed to it.



.. _deepspeed-config-passing:

Passing Configuration
=======================================================================================================================

As discussed in this document normally the DeepSpeed configuration is passed as a path to a json file, but if you're
not using the command line interface to configure the training, and instead instantiate the
:class:`~transformers.Trainer` via :class:`~transformers.TrainingArguments` then for the ``deepspeed`` argument you can
pass a nested ``dict``. This allows you to create the configuration on the fly and doesn't require you to write it to
the file system before passing it to :class:`~transformers.TrainingArguments`.

To summarize you can do:

.. code-block:: python

    TrainingArguments(..., deespeed="/path/to/ds_config.json")

or:

.. code-block:: python

    ds_config_dict=dict(scheduler=scheduler_params, optimizer=optimizer_params)
    TrainingArguments(..., deespeed=ds_config_dict)



.. _deepspeed-config-shared:

Shared Configuration
=======================================================================================================================


.. warning::

    This section is a must-read

Some configuration values are required by both the :class:`~transformers.Trainer` and DeepSpeed to function correctly,
therefore, to prevent conflicting definitions, which could lead to hard to detect errors, we chose to configure those
via the :class:`~transformers.Trainer` command line arguments.

Additionally, some configuration values are derived automatically based on the model's configuration, so instead of
remembering to manually adjust multiple values, it's the best to let the :class:`~transformers.Trainer` do the majority
of configuration for you.

Therefore, in the rest of this guide you will find a special configuration value: ``auto``, which when set will be
automatically replaced with the correct or most efficient value. Please feel free to choose to ignore this
recommendation and set the values explicitly, in which case be very careful that your the
:class:`~transformers.Trainer` arguments and DeepSpeed configurations agree. For example, are you using the same
learning rate, or batch size, or gradient accumulation settings? if these mismatch the training may fail in very
difficult to detect ways. You have been warned.

There are multiple other values that are specific to DeepSpeed-only and those you will have to set manually to suit
your needs.

In your own programs, you can also use the following approach if you'd like to modify the DeepSpeed config as a master
and configure :class:`~transformers.TrainingArguments` based on that. The steps are:

1. Create or load the DeepSpeed configuration to be used as a master configuration
2. Create the :class:`~transformers.TrainingArguments` object based on these values

Do note that some values, such as :obj:`scheduler.params.total_num_steps` are calculated by
:class:`~transformers.Trainer` during ``train``, but you can of course do the math yourself.

.. _deepspeed-zero:

ZeRO
=======================================================================================================================

`Zero Redundancy Optimizer (ZeRO) <https://www.deepspeed.ai/tutorials/zero/>`__ is the workhorse of DeepSpeed. It
support 3 different levels (stages) of optimization. The first one is not quite interesting for scalability purposes,
therefore this document focuses on stages 2 and 3. Stage 3 is further improved by the latest addition of ZeRO-Infinity.
You will find more indepth information in the DeepSpeed documentation.

The ``zero_optimization`` section of the configuration file is the most important part (`docs
<https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training>`__), since that is where you define
which ZeRO stages you want to enable and how to configure them. You will find the explanation for each parameter in the
DeepSpeed docs.

This section has to be configured exclusively via DeepSpeed configuration - the :class:`~transformers.Trainer` provides
no equivalent command line arguments.

Note: currently DeepSpeed doesn't validate parameter names, so if you misspell any, it'll use the default setting for
the parameter that got misspelled. You can watch the DeepSpeed engine start up log messages to see what values it is
going to use.



.. _deepspeed-zero2-config:

ZeRO-2 Config
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The following is an example configuration for ZeRO stage 2:

.. code-block:: json

    {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "allgather_partitions": true,
            "allgather_bucket_size": 5e8,
            "overlap_comm": true,
            "reduce_scatter": true,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": true
        }
    }

**Performance tuning:**

- enabling ``offload_optimizer`` should reduce GPU RAM usage (it requires ``"stage": 2``)
- ``"overlap_comm": true`` trades off increased GPU RAM usage to lower all-reduce latency. ``overlap_comm`` uses 4.5x
  the ``allgather_bucket_size`` and ``reduce_bucket_size`` values. So if they are set to 5e8, this requires a 9GB
  footprint (``5e8 x 2Bytes x 2 x 4.5``). Therefore, if you have a GPU with 8GB or less RAM, to avoid getting
  OOM-errors you will need to reduce those parameters to about ``2e8``, which would require 3.6GB. You will want to do
  the same on larger capacity GPU as well, if you're starting to hit OOM.
- when reducing these buffers you're trading communication speed to avail more GPU RAM. The smaller the buffer size,
  the slower the communication, and the more GPU RAM will be available to other tasks. So if a bigger batch size is
  important, getting a slightly slower training time could be a good trade.



.. _deepspeed-zero3-config:

ZeRO-3 Config
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The following is an example configuration for ZeRO stage 3:

.. code-block:: json

    {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": true
            },
            "overlap_comm": true,
            "contiguous_gradients": true,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_fp16_weights_on_model_save": true
        }
    }

If you are getting OOMs, because your model or activations don't fit into the GPU memory and you have unutilized CPU
memory offloading the optimizer states and parameters to CPU memory with ``"device": "cpu"`` may solve this limitation.
If you don't want to offload to CPU memory, use ``none`` instead of ``cpu`` for the ``device`` entry. Offloading to
NVMe is discussed further down.

Pinned memory is enabled with ``pin_memory`` set to ``true``. This feature can improve the throughput at the cost of
making less memory available to other processes. Pinned memory is set aside to the specific process that requested it
and its typically accessed much faster than normal CPU memory.

**Performance tuning:**

- ``stage3_max_live_parameters``: ``1e9``
- ``stage3_max_reuse_distance``: ``1e9``

If hitting OOM reduce ``stage3_max_live_parameters`` and ``stage3_max_reuse_distance``. They should have minimal impact
on performance unless you are doing activation checkpointing. ``1e9`` would consume ~2GB. The memory is shared by
``stage3_max_live_parameters`` and ``stage3_max_reuse_distance``, so its not additive, its just 2GB total.

``stage3_max_live_parameters`` is the upper limit on how many full parameters you want to keep on the GPU at any given
time. "reuse distance" is a metric we are using to figure out when will a parameter be used again in the future, and we
use the ``stage3_max_reuse_distance`` to decide whether to throw away the parameter or to keep it. If a parameter is
going to be used again in near future (less than ``stage3_max_reuse_distance``) then we keep it to reduce communication
overhead. This is super helpful when you have activation checkpointing enabled, where we do a forward recompute and
backward passes a a single layer granularity and want to keep the parameter in the forward recompute till the backward

The following configuration values depend on the model's hidden size:

- ``reduce_bucket_size``: ``hidden_size*hidden_size``
- ``stage3_prefetch_bucket_size``: ``0.9 * hidden_size * hidden_size``
- ``stage3_param_persistence_threshold``: ``10 * hidden_size``

therefore set these values to ``auto`` and the :class:`~transformers.Trainer` will automatically assign the recommended
values. But, of course, feel free to set these explicitly as well.

``stage3_gather_fp16_weights_on_model_save`` enables model fp16 weights consolidation when model gets saved. With large
models and multiple GPUs this is an expensive operation both in terms of memory and speed. It's currently required if
you plan to resume the training. Watch out for future updates that will remove this limitation and make things more
flexible.

If you're migrating from ZeRO-2 configuration note that ``allgather_partitions``, ``allgather_bucket_size`` and
``reduce_scatter`` configuration parameters are not used in ZeRO-3. If you keep these in the config file they will just
be ignored.

- ``sub_group_size``: ``1e9``

``sub_group_size`` controls the granularity in which parameters are updated during optimizer steps. Parameters are
grouped into buckets of ``sub_group_size`` and each buckets is updated one at a time. When used with NVMe offload in
ZeRO-Infinity, ``sub_group_size`` therefore controls the granularity in which model states are moved in and out of CPU
memory from NVMe during the optimizer step. This prevents running out of CPU memory for extremely large models.

You can leave ``sub_group_size`` to its default value of `1e9` when not using NVMe offload. You may want to change its
default value in the following cases:

1. Running into OOM during optimizer step: Reduce ``sub_group_size`` to reduce memory utilization of temporary buffers
2. Optimizer Step is taking a long time: Increase ``sub_group_size`` to improve bandwidth utilization as a result of
   the increased data buffers.


.. _deepspeed-nvme:

NVMe Support
=======================================================================================================================

ZeRO-Infinity allows for training incredibly large models by extending GPU and CPU memory with NVMe memory. Thanks to
smart partitioning and tiling algorithms each GPU needs to send and receive very small amounts of data during
offloading so modern NVMe proved to be fit to allow for an even larger total memory pool available to your training
process. ZeRO-Infinity requires ZeRO-3 enabled.

The following configuration example enables NVMe to offload both optimizer states and the params:

.. code-block:: json

    {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "nvme",
                "nvme_path": "/local_nvme",
                "pin_memory": true,
                "buffer_count": 4,
                "fast_init": false
            },
            "offload_param": {
                "device": "nvme",
                "nvme_path": "/local_nvme",
                "pin_memory": true,
                "buffer_count": 5,
                "buffer_size": 1e8,
                "max_in_cpu": 1e9
            }
            "aio": {
                "block_size": 262144,
                "queue_depth": 32,
                "thread_count": 1,
                "single_submit": false,
                "overlap_events": true
            }
            "overlap_comm": true,
            "contiguous_gradients": true,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_fp16_weights_on_model_save": true
        },
    }

You can choose to offload both optimizer states and params to NVMe, or just one of them or none. For example, if you
have copious amounts of CPU memory available, by all means offload to CPU memory only as it'd be faster (hint:
`"device": "cpu"`).

Here is the full documentation for offloading `optimizer states
<https://www.deepspeed.ai/docs/config-json/#optimizer-offloading>`__ and `parameters
<https://www.deepspeed.ai/docs/config-json/#parameter-offloading>`__.

Make sure that your ``nvme_path`` is actually an NVMe, since it will work with the normal hard drive or SSD, but it'll
be much much slower. The fast scalable training was designed with modern NVMe transfer speeds in mind (as of this
writing one can have ~3.5GB/s read, ~3GB/s write peak speeds).

In order to figure out the optimal ``aio`` configuration block you must run a benchmark on your target setup, as
`explained here <https://github.com/microsoft/DeepSpeed/issues/998>`__.



.. _deepspeed-zero2-zero3-performance:

ZeRO-2 vs ZeRO-3 Performance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ZeRO-3 is likely to be slower than ZeRO-2 if everything else is configured the same because the former has to gather
model weights in addition to what ZeRO-2 does. If ZeRO-2 meets your needs and you don't need to scale beyond a few GPUs
then you may choose to stick to it. It's important to understand that ZeRO-3 enables a much higher scalability capacity
at a cost of speed.

It's possible to adjust ZeRO-3 configuration to make it perform closer to ZeRO-2:

- set ``stage3_param_persistence_threshold`` to a very large number - larger than the largest parameter, e.g., ``6 *
  hidden_size * hidden_size``. This will keep the parameters on the GPUs.
- turn off ``offload_params`` since ZeRO-2 doesn't have that option.

The performance will likely improve significantly with just ``offload_params`` turned off, even if you don't change
``stage3_param_persistence_threshold``. Of course, these changes will impact the size of the model you can train. So
these help you to trade scalability for speed depending on your needs.



.. _deepspeed-zero2-example:

ZeRO-2 Example
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Here is a full ZeRO-2 auto-configuration file ``ds_config_zero2.json``:

.. code-block:: json

    {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },

        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "allgather_partitions": true,
            "allgather_bucket_size": 2e8,
            "overlap_comm": true,
            "reduce_scatter": true,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": true
        },

        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": false
    }


Here is a full ZeRO-2 all-enabled manually set configuration file. It is here mainly for you to see what the typical
values look like, but we highly recommend using the one with multiple ``auto`` settings in it.

.. code-block:: json

    {
        "fp16": {
            "enabled": true,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-5,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 500
            }
        },

        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "allgather_partitions": true,
            "allgather_bucket_size": 2e8,
            "overlap_comm": true,
            "reduce_scatter": true,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": true
        },

        "steps_per_print": 2000,
        "wall_clock_breakdown": false
    }



.. _deepspeed-zero3-example:

ZeRO-3 Example
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Here is a full ZeRO-3 auto-configuration file ``ds_config_zero3.json``:


.. code-block:: json

    {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },

        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": true
            },
            "overlap_comm": true,
            "contiguous_gradients": true,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_fp16_weights_on_model_save": true
        },

        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": false
    }

Here is a full ZeRO-3 all-enabled manually set configuration file. It is here mainly for you to see what the typical
values look like, but we highly recommend using the one with multiple ``auto`` settings in it.

.. code-block:: json

    {
        "fp16": {
            "enabled": true,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-5,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },

        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 500
            }
        },

        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": true
            },
            "overlap_comm": true,
            "contiguous_gradients": true,
            "sub_group_size": 1e9,
            "reduce_bucket_size": 1e6,
            "stage3_prefetch_bucket_size": 0.94e6,
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_fp16_weights_on_model_save": true
        },

        "steps_per_print": 2000,
        "wall_clock_breakdown": false
    }


Optimizer and Scheduler
=======================================================================================================================

As long as you don't enable ``offload_optimizer`` you can mix and match DeepSpeed and HuggingFace schedulers and
optimizers, with the exception of using the combination of HuggingFace scheduler and DeepSpeed optimizer:

+--------------+--------------+--------------+
| Combos       | HF Scheduler | DS Scheduler |
+--------------+--------------+--------------+
| HF Optimizer | Yes          | Yes          |
+--------------+--------------+--------------+
| DS Optimizer | No           | Yes          |
+--------------+--------------+--------------+

It is possible to use a non-DeepSpeed optimizer when ``offload_optimizer`` is enabled, as long as it has both CPU and
GPU implementation (except LAMB).




.. _deepspeed-optimizer:

Optimizer
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


DeepSpeed's main optimizers are Adam, AdamW, OneBitAdam, and Lamb. These have been thoroughly tested with ZeRO and are
thus recommended to be used. It, however, can import other optimizers from ``torch``. The full documentation is `here
<https://www.deepspeed.ai/docs/config-json/#optimizer-parameters>`__.

If you don't configure the ``optimizer`` entry in the configuration file, the :class:`~transformers.Trainer` will
automatically set it to ``AdamW`` and will use the supplied values or the defaults for the following command line
arguments: ``--learning_rate``, ``--adam_beta1``, ``--adam_beta2``, ``--adam_epsilon`` and ``--weight_decay``.

Here is an example of the auto-configured ``optimizer`` entry for ``AdamW``:

.. code-block:: json

    {
       "optimizer": {
           "type": "AdamW",
           "params": {
             "lr": "auto",
             "betas": "auto",
             "eps": "auto",
             "weight_decay": "auto"
           }
       }
    }


Note that the command line arguments will set the values in the configuration file. This is so that there is one
definitive source of the values and to avoid hard to find errors when for example, the learning rate is set to
different values in different places. Command line rules. The values that get overridden are:

- ``lr`` with the value of ``--learning_rate``
- ``betas`` with the value of ``--adam_beta1 --adam_beta2``
- ``eps`` with the value of ``--adam_epsilon``
- ``weight_decay`` with the value of ``--weight_decay``

Therefore please remember to tune the shared hyperparameters on the command line.

You can also set the values explicitly:

.. code-block:: json

    {
       "optimizer": {
           "type": "AdamW",
           "params": {
             "lr": 0.001,
             "betas": [0.8, 0.999],
             "eps": 1e-8,
             "weight_decay": 3e-7
           }
       }
    }

But then you're on your own synchronizing the :class:`~transformers.Trainer` command line arguments and the DeepSpeed
configuration.

If you want to use another optimizer which is not listed above, you will have to add to the top level configuration.

.. code-block:: json

    {
       "zero_allow_untested_optimizer": true
    }

Similarly to ``AdamW``, you can configure other officially supported optimizers. Just remember that may have different
config values. e.g. for Adam you will want ``weight_decay`` around ``0.01``.



.. _deepspeed-scheduler:

Scheduler
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

DeepSpeed supports ``LRRangeTest``, ``OneCycle``, ``WarmupLR`` and ``WarmupDecayLR`` learning rate schedulers. The full
documentation is `here <https://www.deepspeed.ai/docs/config-json/#scheduler-parameters>`__.

Here is where the schedulers overlap between 🤗 Transformers and DeepSpeed:

* ``WarmupLR`` via ``--lr_scheduler_type constant_with_warmup``
* ``WarmupDecayLR`` via ``--lr_scheduler_type linear``. This is also the default value for ``--lr_scheduler_type``,
  therefore, if you don't configure the scheduler this is scheduler that will get configured by default.

If you don't configure the ``scheduler`` entry in the configuration file, the :class:`~transformers.Trainer` will use
the values of ``--lr_scheduler_type``, ``--learning_rate`` and ``--warmup_steps`` or ``--warmup_ratio`` to configure a
🤗 Transformers version of it.

Here is an example of the auto-configured ``scheduler`` entry for ``WarmupLR``:

.. code-block:: json

    {
       "scheduler": {
             "type": "WarmupLR",
             "params": {
                 "warmup_min_lr": "auto",
                 "warmup_max_lr": "auto",
                 "warmup_num_steps": "auto"
             }
         }
    }

Since `"auto"` is used the :class:`~transformers.Trainer` arguments will set the correct values in the configuration
file. This is so that there is one definitive source of the values and to avoid hard to find errors when, for example,
the learning rate is set to different values in different places. Command line rules. The values that get set are:

- ``warmup_min_lr`` with the value of ``0``.
- ``warmup_max_lr`` with the value of ``--learning_rate``.
- ``warmup_num_steps`` with the value of ``--warmup_steps`` if provided. Otherwise will use ``--warmup_ratio``
  multiplied by the number of training steps and rounded up.
- ``total_num_steps`` with either the value of ``--max_steps`` or if it is not provided, derived automatically at run
  time based on the environment and the size of the dataset and other command line arguments (needed for
  ``WarmupDecayLR``).

You can, of course, take over any or all of the configuration values and set those yourself:

.. code-block:: json

    {
       "scheduler": {
             "type": "WarmupLR",
             "params": {
                 "warmup_min_lr": 0,
                 "warmup_max_lr": 0.001,
                 "warmup_num_steps": 1000
             }
         }
    }

But then you're on your own synchronizing the :class:`~transformers.Trainer` command line arguments and the DeepSpeed
configuration.

For example, for ``WarmupDecayLR``, you can use the following entry:

.. code-block:: json

    {
       "scheduler": {
             "type": "WarmupDecayLR",
             "params": {
                 "last_batch_iteration": -1,
                 "total_num_steps": "auto",
                 "warmup_min_lr": "auto",
                 "warmup_max_lr": "auto",
                 "warmup_num_steps": "auto"
             }
         }
    }

and ``total_num_steps`, ``warmup_max_lr``, ``warmup_num_steps`` and ``total_num_steps`` will be set at loading time.




.. _deepspeed-fp32:

fp32 Precision
=======================================================================================================================

Deepspeed supports the full fp32 and the fp16 mixed precision.

Because of the much reduced memory needs and faster speed one gets with the fp16 mixed precision, the only time you
will want to not use it is when the model you're using doesn't behave well under this training mode. Typically this
happens when the model wasn't pretrained in the fp16 mixed precision (e.g. often this happens with bf16-pretrained
models). Such models may overflow or underflow leading to ``NaN`` loss. If this is your case then you will want to use
the full fp32 mode, by explicitly disabling the otherwise default fp16 mixed precision mode with:

.. code-block:: json

    {
        "fp16": {
            "enabled": "false",
        }
    }

If you're using the Ampere-architecture based GPU, pytorch version 1.7 and higher will automatically switch to using
the much more efficient tf32 format for some operations, but the results will still be in fp32. For details and
benchmarks, please, see `TensorFloat-32(TF32) on Ampere devices
<https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices>`__. The document includes
instructions on how to disable this automatic conversion if for some reason you prefer not to use it.




.. _deepspeed-amp:

Automatic Mixed Precision
=======================================================================================================================

You can use automatic mixed precision with either a pytorch-like AMP way or the apex-like way:

To configure pytorch AMP-like mode set:

.. code-block:: json

    {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    }

and the :class:`~transformers.Trainer` will automatically enable or disable it based on the value of
``args.fp16_backend``. The rest of config values are up to you.

This mode gets enabled when ``--fp16 --fp16_backend amp`` command line args are passed.

You can also enable/disable this mode explicitly:

.. code-block:: json

    {
        "fp16": {
            "enabled": true,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    }

But then you're on your own synchronizing the :class:`~transformers.Trainer` command line arguments and the DeepSpeed
configuration.

Here is the `documentation <https://www.deepspeed.ai/docs/config-json/#fp16-training-options>`__.

To configure apex AMP-like mode set:

.. code-block:: json

    "amp": {
        "enabled": "auto",
        "opt_level": "auto"
    }

and the :class:`~transformers.Trainer` will automatically configure it based on the values of ``args.fp16_backend`` and
``args.fp16_opt_level``.

This mode gets enabled when ``--fp16 --fp16_backend apex --fp16_opt_level 01`` command line args are passed.

You can also configure this mode explicitly:

.. code-block:: json

    {
        "amp": {
            "enabled": true,
            "opt_level": "O1"
        }
    }

But then you're on your own synchronizing the :class:`~transformers.Trainer` command line arguments and the DeepSpeed
configuration.

Here is the `documentation
<https://www.deepspeed.ai/docs/config-json/#automatic-mixed-precision-amp-training-options>`__.



.. _deepspeed-bs:

Batch Size
=======================================================================================================================

To configure batch size, use:

.. code-block:: json

    {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto"
    }

and the :class:`~transformers.Trainer` will automatically set ``train_micro_batch_size_per_gpu`` to the value of
``args.per_device_train_batch_size`` and ``train_batch_size`` to ``args.world_size * args.per_device_train_batch_size *
args.gradient_accumulation_steps``.

You can also set the values explicitly:

.. code-block:: json

    {
        "train_batch_size": 12,
        "train_micro_batch_size_per_gpu": 4
    }

But then you're on your own synchronizing the :class:`~transformers.Trainer` command line arguments and the DeepSpeed
configuration.



.. _deepspeed-grad-acc:

Gradient Accumulation
=======================================================================================================================

To configure gradient accumulation set:

.. code-block:: json

    {
        "gradient_accumulation_steps": "auto"
    }

and the :class:`~transformers.Trainer` will automatically set it to the value of ``args.gradient_accumulation_steps``.

You can also set the value explicitly:

.. code-block:: json

    {
        "gradient_accumulation_steps": 3
    }

But then you're on your own synchronizing the :class:`~transformers.Trainer` command line arguments and the DeepSpeed
configuration.



.. _deepspeed-grad-clip:

Gradient Clipping
=======================================================================================================================

To configure gradient gradient clipping set:

.. code-block:: json

    {
        "gradient_clipping": "auto"
    }

and the :class:`~transformers.Trainer` will automatically set it to the value of ``args.max_grad_norm``.

You can also set the value explicitly:

.. code-block:: json

    {
        "gradient_clipping": 1.0
    }

But then you're on your own synchronizing the :class:`~transformers.Trainer` command line arguments and the DeepSpeed
configuration.



.. _deepspeed-weight-extraction:

Getting The Model Weights Out
=======================================================================================================================

As long as you continue training and resuming using DeepSpeed you don't need to worry about anything. DeepSpeed stores
fp32 master weights in its custom checkpoint optimizer files, which are ``global_step*/*optim_states.pt`` (this is glob
pattern), and are saved under the normal checkpoint.

**FP16 Weights:**

When a model is saved under ZeRO-2, you end up having the normal ``pytorch_model.bin`` file with the model weights, but
they are only the fp16 version of the weights.

Under ZeRO-3, things are much more complicated, since the model weights are partitioned out over multiple GPUs,
therefore ``"stage3_gather_fp16_weights_on_model_save": true`` is required to get the ``Trainer`` to save the fp16
version of the weights. If this setting is ``False`` ``pytorch_model.bin`` won't be created. This is because by default
DeepSpeed's ``state_dict`` contains a placeholder and not the real weights. If we were to save this ``state_dict`` it
won't be possible to load it back.


.. code-block:: json

    {
        "zero_optimization": {
            "stage3_gather_fp16_weights_on_model_save": true
        }
    }


**FP32 Weights:**

While the fp16 weights are fine for resuming training, if you finished finetuning your model and want to upload it to
the `models hub <https://huggingface.co/models>`__ or pass it to someone else you most likely will want to get the fp32
weights. This ideally shouldn't be done during training since this is a process that requires a lot of memory, and
therefore best to be performed offline after the training is complete. But if desired and you have plenty of free CPU
memory it can be done in the same training script. The following sections will discuss both approaches.


**Live FP32 Weights Recovery:**

This approach may not work if you model is large and you have little free CPU memory left, at the end of the training.

If you have saved at least one checkpoint, and you want to use the latest one, you can do the following:

.. code-block:: python

    from transformers.trainer_utils import get_last_checkpoint
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    checkpoint_dir = get_last_checkpoint(trainer.args.output_dir)
    fp32_model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)

If you're using the ``--load_best_model_at_end`` class:`~transformers.TrainingArguments` argument (to track the best
checkpoint), then you can finish the training by first saving the final model explicitly and then do the same as above:

.. code-block:: python

    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    checkpoint_dir = os.path.join(trainer.args.output_dir, "checkpoint-final")
    trainer.deepspeed.save_checkpoint(checkpoint_dir)
    fp32_model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)

.. note::

    Note, that once ``load_state_dict_from_zero_checkpoint`` was run, the ``model`` will no longer be useable in the
    DeepSpeed context of the same application. i.e. you will need to re-initialize the deepspeed engine, since
    ``model.load_state_dict(state_dict)`` will remove all the DeepSpeed magic from it. So do this only at the very end
    of the training.

Of course, you don't have to use class:`~transformers.Trainer` and you can adjust the examples above to your own
trainer.

If for some reason you want more refinement, you can also extract the fp32 ``state_dict`` of the weights and apply
these yourself as is shown in the following example:

.. code-block:: python

    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir) # already on cpu
    model = model.cpu()
    model.load_state_dict(state_dict)


**Offline FP32 Weights Recovery:**

DeepSpeed creates a special conversion script ``zero_to_fp32.py`` which it places in the top-level of the checkpoint
folder. Using this script you can extract the weights at any point. The script is standalone and you no longer need to
have the configuration file or a ``Trainer`` to do the extraction.

Let's say your checkpoint folder looks like this:

.. code-block:: bash

    $ ls -l output_dir/checkpoint-1/
    -rw-rw-r-- 1 stas stas 1.4K Mar 27 20:42 config.json
    drwxrwxr-x 2 stas stas 4.0K Mar 25 19:52 global_step1/
    -rw-rw-r-- 1 stas stas   12 Mar 27 13:16 latest
    -rw-rw-r-- 1 stas stas 827K Mar 27 20:42 optimizer.pt
    -rw-rw-r-- 1 stas stas 231M Mar 27 20:42 pytorch_model.bin
    -rw-rw-r-- 1 stas stas  623 Mar 27 20:42 scheduler.pt
    -rw-rw-r-- 1 stas stas 1.8K Mar 27 20:42 special_tokens_map.json
    -rw-rw-r-- 1 stas stas 774K Mar 27 20:42 spiece.model
    -rw-rw-r-- 1 stas stas 1.9K Mar 27 20:42 tokenizer_config.json
    -rw-rw-r-- 1 stas stas  339 Mar 27 20:42 trainer_state.json
    -rw-rw-r-- 1 stas stas 2.3K Mar 27 20:42 training_args.bin
    -rwxrw-r-- 1 stas stas 5.5K Mar 27 13:16 zero_to_fp32.py*

In this example there is just one DeepSpeed checkpoint sub-folder `global_step1`. Therefore to reconstruct the fp32
weights just run:

.. code-block:: bash

    python zero_to_fp32.py . pytorch_model.bin

This is it. ``pytorch_model.bin`` will now contain the full fp32 model weights consolidated from multiple GPUs.

The script will automatically be able to handle either a ZeRO-2 or ZeRO-3 checkpoint.

``python zero_to_fp32.py -h`` will give you usage details.

The script will auto-discover the deepspeed sub-folder using the contents of the file ``latest``, which in the current
example will contain ``global_step1``.

Note: currently the script requires 2x general RAM of the final fp32 model weights.


ZeRO-3 and Infinity Nuances
=======================================================================================================================

ZeRO-3 is quite different from ZeRO-2 because of its param sharding feature.

ZeRO-Infinity further extends ZeRO-3 to support NVMe memory and multiple other speed and scalability improvements.

While all the efforts were made for things to just work without needing any special changes to your models, in certain
circumstances you may find the following information to be needed.



Constructing Massive Models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

DeepSpeed/ZeRO-3 can handle models with Trillions of parameters which may not fit onto the existing RAM. In such cases,
but also if you want the initialization to happen much faster, initialize the model using `deepspeed.zero.Init()`
context manager (which is also a function decorator), like so:

.. code-block:: python

    from transformers import T5ForConditionalGeneration, T5Config
    import deepspeed
    with deepspeed.zero.Init():
       config = T5Config.from_pretrained("t5-small")
       model = T5ForConditionalGeneration(config)

As you can see this gives you a randomly initialized model.

If you want to use a pretrained model, ``model_class.from_pretrained`` will activate this feature as long as
``is_deepspeed_zero3_enabled()`` returns ``True``, which currently is setup by the
class:`~transformers.TrainingArguments` object if the passed DeepSpeed configuration file contains ZeRO-3 config
section. Thus you must create the :class:`~transformers.TrainingArguments` object **before** calling
``from_pretrained``. Here is an example of a possible sequence:

.. code-block:: python

    from transformers import AutoModel, Trainer, TrainingArguments
    training_args = TrainingArguments(..., deepspeed=ds_config)
    model = AutoModel.from_pretrained("t5-small")
    trainer = Trainer(model=model, args=training_args, ...)

If you're using the official example scripts and your command line arguments include ``--deepspeed ds_config.json``
with ZeRO-3 config enabled, then everything is already done for you, since this is how example scripts are written.

Note: If the fp16 weights of the model can't fit onto the memory of a single GPU this feature must be used.

For full details on this method and other related features please refer to `Constructing Massive Models
<https://deepspeed.readthedocs.io/en/latest/zero3.html#constructing-massive-models>`__.

Also when loading fp16-pretrained models, you will want to tell ``from_pretrained`` to use
``torch_dtype=torch.float16``. For details, please, see :ref:`from_pretrained-torch-dtype`.


Gathering Parameters
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Under ZeRO-3 on multiple GPUs no single GPU has all the parameters unless it's the parameters for the currently
executing layer. So if you need to access all parameters from all layers at once there is a specific method to do it.
Most likely you won't need it, but if you do please refer to `Gathering Parameters
<https://deepspeed.readthedocs.io/en/latest/zero3.html#manual-parameter-coordination>`__

We do however use it internally in several places, one such example is when loading pretrained model weights in
``from_pretrained``. We load one layer at a time and immediately partition it to all participating GPUs, as for very
large models it won't be possible to load it on one GPU and then spread it out to multiple GPUs, due to memory
limitations.

Also under ZeRO-3, if you write your own code and run into a model parameter weight that looks like:

.. code-block:: python

    tensor([1.], device='cuda:0', dtype=torch.float16, requires_grad=True)

stress on ``tensor([1.])``, or if you get an error where it says the parameter is of size ``1``, instead of some much
larger multi-dimensional shape, this means that the parameter is partitioned and what you see is a ZeRO-3 placeholder.



.. _deepspeed-zero-inference:


ZeRO Inference
=======================================================================================================================

ZeRO Inference uses the same config as ZeRO-3 Training. You just don't need the optimizer and scheduler sections. In
fact you can leave these in the config file if you want to share the same one with the training. They will just be
ignored.

Otherwise you just need to pass the usual :class:`~transformers.TrainingArguments` arguments. For example:

.. code-block:: bash

    deepspeed --num_gpus=2 your_program.py <normal cl args> --do_eval --deepspeed ds_config.json

The only important thing is that you need to use a ZeRO-3 configuration, since ZeRO-2 provides no benefit whatsoever
for the inference as only ZeRO-3 performs sharding of parameters, whereas ZeRO-1 shards gradients and optimizer states.

Here is an example of running ``run_translation.py`` under DeepSpeed deploying all available GPUs:

.. code-block:: bash

    deepspeed examples/pytorch/translation/run_translation.py \
    --deepspeed tests/deepspeed/ds_config_zero3.json \
    --model_name_or_path t5-small --output_dir output_dir \
    --do_eval --max_eval_samples 50 --warmup_steps 50  \
    --max_source_length 128 --val_max_target_length 128 \
    --overwrite_output_dir --per_device_eval_batch_size 4 \
    --predict_with_generate --dataset_config "ro-en" --fp16 \
    --source_lang en --target_lang ro --dataset_name wmt16 \
    --source_prefix "translate English to Romanian: "

Since for inference there is no need for additional large memory used by the optimizer states and the gradients you
should be able to fit much larger batches and/or sequence length onto the same hardware.


Additionally DeepSpeed is currently developing a related product called Deepspeed-Inference which has no relationship
to the ZeRO technology, but instead uses tensor parallelism to scale models that can't fit onto a single GPU. This is a
work in progress and we will provide the integration once that product is complete.


Filing Issues
=======================================================================================================================

Here is how to file an issue so that we could quickly get to the bottom of the issue and help you to unblock your work.

In your report please always include:

1. the full Deepspeed config file in the report

2. either the command line arguments if you were using the :class:`~transformers.Trainer` or
   :class:`~transformers.TrainingArguments` arguments if you were scripting the Trainer setup yourself. Please do not
   dump the :class:`~transformers.TrainingArguments` as it has dozens of entries that are irrelevant.

3. Output of:

.. code-block:: bash

    python -c 'import torch; print(f"torch: {torch.__version__}")'
    python -c 'import transformers; print(f"transformers: {transformers.__version__}")'
    python -c 'import deepspeed; print(f"deepspeed: {deepspeed.__version__}")'

4. If possible include a link to a Google Colab notebook that we can reproduce the problem with. You can use this
   `notebook <https://github.com/stas00/porting/blob/master/transformers/deepspeed/DeepSpeed_on_colab_CLI.ipynb>`__ as
   a starting point.

5. Unless it's impossible please always use a standard dataset that we can use and not something custom.

6. If possible try to use one of the existing `examples
   <https://github.com/huggingface/transformers/tree/master/examples/pytorch>`__ to reproduce the problem with.

Things to consider:

* Deepspeed is often not the cause of the problem.

    Some of the filed issues proved to be Deepspeed-unrelated. That is once Deepspeed was removed from the setup, the
    problem was still there.

    Therefore, if it's not absolutely obvious it's a DeepSpeed-related problem, as in you can see that there is an
    exception and you can see that DeepSpeed modules are involved, first re-test your setup without DeepSpeed in it.
    And only if the problem persists then do mentioned Deepspeed and supply all the required details.

* If it's clear to you that the issue is in the DeepSpeed core and not the integration part, please file the Issue
  directly with `Deepspeed <https://github.com/microsoft/DeepSpeed/>`__. If you aren't sure, please do not worry,
  either Issue tracker will do, we will figure it out once you posted it and redirect you to another Issue tracker if
  need be.



Troubleshooting
=======================================================================================================================

* ``deepspeed`` process gets killed at startup without a traceback

If the ``deepspeed`` process gets killed at launch time without a traceback, that usually means that the program tried
to allocate more CPU memory than your system has or your process is allowed to allocate and the OS kernel killed that
process. This is because your configuration file most likely has either ``offload_optimizer`` or ``offload_param`` or
both configured to offload to ``cpu``. If you have NVMe, experiment with offloading to NVMe if you're running under
ZeRO-3.

Work is being done to enable estimating how much memory is needed for a specific model: `PR
<https://github.com/microsoft/DeepSpeed/pull/965>`__.






Notes
=======================================================================================================================

* DeepSpeed works with the PyTorch :class:`~transformers.Trainer` but not TF :class:`~transformers.TFTrainer`.
* While DeepSpeed has a pip installable PyPI package, it is highly recommended that it gets installed from `source
  <https://github.com/microsoft/deepspeed#installation>`__ to best match your hardware and also if you need to enable
  certain features, like 1-bit Adam, which aren't available in the pypi distribution.
* You don't have to use the :class:`~transformers.Trainer` to use DeepSpeed with 🤗 Transformers - you can use any model
  with your own trainer, and you will have to adapt the latter according to `the DeepSpeed integration instructions
  <https://www.deepspeed.ai/getting-started/#writing-deepspeed-models>`__.




.. _deepspeed-non-trainer-integration:

Non-Trainer Deepspeed Integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~transformers.integrations.HfDeepSpeedConfig` is used to integrate Deepspeed into the 🤗 Transformers core
functionality, when :class:`~transformers.Trainer` is not used.

When using :class:`~transformers.Trainer` everything is automatically taken care of.

When not using :class:`~transformers.Trainer`, to efficiently deploy DeepSpeed stage 3, you must instantiate the
:class:`~transformers.integrations.HfDeepSpeedConfig` object before instantiating the model.

For example for a pretrained model:

.. code-block:: python

    from transformers.deepspeed import HfDeepSpeedConfig
    from transformers import AutoModel, deepspeed

    ds_config = { ... } # deepspeed config object or path to the file
    # must run before instantiating the model
    dschf = HfDeepSpeedConfig(ds_config) # keep this object alive
    model = AutoModel.from_pretrained("gpt2")
    engine = deepspeed.initialize(model=model, config_params=ds_config, ...)

or for non-pretrained model:

.. code-block:: python

    from transformers.deepspeed import HfDeepSpeedConfig
    from transformers import AutoModel, AutoConfig, deepspeed

    ds_config = { ... } # deepspeed config object or path to the file
    # must run before instantiating the model
    dschf = HfDeepSpeedConfig(ds_config) # keep this object alive
    config = AutoConfig.from_pretrained("gpt2")
    model = AutoModel.from_config(config)
    engine = deepspeed.initialize(model=model, config_params=ds_config, ...)


HfDeepSpeedConfig
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: transformers.deepspeed.HfDeepSpeedConfig
    :members:



Main DeepSpeed Resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Project's github <https://github.com/microsoft/deepspeed>`__
- `Usage docs <https://www.deepspeed.ai/getting-started/>`__
- `API docs <https://deepspeed.readthedocs.io/en/latest/index.html>`__
- `Blog posts <https://www.microsoft.com/en-us/research/search/?q=deepspeed>`__

Papers:

- `ZeRO: Memory Optimizations Toward Training Trillion Parameter Models <https://arxiv.org/abs/1910.02054>`__
- `ZeRO-Offload: Democratizing Billion-Scale Model Training <https://arxiv.org/abs/2101.06840>`__
- `ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning <https://arxiv.org/abs/2104.07857>`__

Finally, please, remember that, HuggingFace :class:`~transformers.Trainer` only integrates DeepSpeed, therefore if you
have any problems or questions with regards to DeepSpeed usage, please, file an issue with `DeepSpeed GitHub
<https://github.com/microsoft/DeepSpeed/issues>`__.
