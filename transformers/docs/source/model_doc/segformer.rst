.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

SegFormer
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SegFormer model was proposed in `SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
<https://arxiv.org/abs/2105.15203>`__ by Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping
Luo. The model consists of a hierarchical Transformer encoder and a lightweight all-MLP decode head to achieve great
results on image segmentation benchmarks such as ADE20K and Cityscapes.

The abstract from the paper is the following:

*We present SegFormer, a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with
lightweight multilayer perception (MLP) decoders. SegFormer has two appealing features: 1) SegFormer comprises a novel
hierarchically structured Transformer encoder which outputs multiscale features. It does not need positional encoding,
thereby avoiding the interpolation of positional codes which leads to decreased performance when the testing resolution
differs from training. 2) SegFormer avoids complex decoders. The proposed MLP decoder aggregates information from
different layers, and thus combining both local attention and global attention to render powerful representations. We
show that this simple and lightweight design is the key to efficient segmentation on Transformers. We scale our
approach up to obtain a series of models from SegFormer-B0 to SegFormer-B5, reaching significantly better performance
and efficiency than previous counterparts. For example, SegFormer-B4 achieves 50.3% mIoU on ADE20K with 64M parameters,
being 5x smaller and 2.2% better than the previous best method. Our best model, SegFormer-B5, achieves 84.0% mIoU on
Cityscapes validation set and shows excellent zero-shot robustness on Cityscapes-C.*

This model was contributed by `nielsr <https://huggingface.co/nielsr>`__. The original code can be found `here
<https://github.com/NVlabs/SegFormer>`__.

The figure below illustrates the architecture of SegFormer. Taken from the `original paper
<https://arxiv.org/abs/2105.15203>`__.

.. image:: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/segformer_architecture.png
  :width: 600

Tips:

- SegFormer consists of a hierarchical Transformer encoder, and a lightweight all-MLP decode head.
  :class:`~transformers.SegformerModel` is the hierarchical Transformer encoder (which in the paper is also referred to
  as Mix Transformer or MiT). :class:`~transformers.SegformerForSemanticSegmentation` adds the all-MLP decode head on
  top to perform semantic segmentation of images. In addition, there's
  :class:`~transformers.SegformerForImageClassification` which can be used to - you guessed it - classify images. The
  authors of SegFormer first pre-trained the Transformer encoder on ImageNet-1k to classify images. Next, they throw
  away the classification head, and replace it by the all-MLP decode head. Next, they fine-tune the model altogether on
  ADE20K, Cityscapes and COCO-stuff, which are important benchmarks for semantic segmentation. All checkpoints can be
  found on the `hub <https://huggingface.co/models?other=segformer>`__.
- The quickest way to get started with SegFormer is by checking the `example notebooks
  <https://github.com/NielsRogge/Transformers-Tutorials/tree/master/SegFormer>`__ (which showcase both inference and
  fine-tuning on custom data).
- One can use :class:`~transformers.SegformerFeatureExtractor` to prepare images and corresponding segmentation maps
  for the model. Note that this feature extractor is fairly basic and does not include all data augmentations used in
  the original paper. The original preprocessing pipelines (for the ADE20k dataset for instance) can be found `here
  <https://github.com/NVlabs/SegFormer/blob/master/local_configs/_base_/datasets/ade20k_repeat.py>`__. The most
  important preprocessing step is that images and segmentation maps are randomly cropped and padded to the same size,
  such as 512x512 or 640x640, after which they are normalized.
- One additional thing to keep in mind is that one can initialize :class:`~transformers.SegformerFeatureExtractor` with
  :obj:`reduce_labels` set to `True` or `False`. In some datasets (like ADE20k), the 0 index is used in the annotated
  segmentation maps for background. However, ADE20k doesn't include the "background" class in its 150 labels.
  Therefore, :obj:`reduce_labels` is used to reduce all labels by 1, and to make sure no loss is computed for the
  background class (i.e. it replaces 0 in the annotated maps by 255, which is the `ignore_index` of the loss function
  used by :class:`~transformers.SegformerForSemanticSegmentation`). However, other datasets use the 0 index as
  background class and include this class as part of all labels. In that case, :obj:`reduce_labels` should be set to
  `False`, as loss should also be computed for the background class.
- As most models, SegFormer comes in different sizes, the details of which can be found in the table below.

+-------------------+---------------+---------------------+-------------------------+----------------+-----------------------+
| **Model variant** | **Depths**    | **Hidden sizes**    | **Decoder hidden size** | **Params (M)** | **ImageNet-1k Top 1** |
+-------------------+---------------+---------------------+-------------------------+----------------+-----------------------+
| MiT-b0            | [2, 2, 2, 2]  | [32, 64, 160, 256]  | 256                     | 3.7            | 70.5                  |
+-------------------+---------------+---------------------+-------------------------+----------------+-----------------------+
| MiT-b1            | [2, 2, 2, 2]  | [64, 128, 320, 512] | 256                     | 14.0           | 78.7                  |
+-------------------+---------------+---------------------+-------------------------+----------------+-----------------------+
| MiT-b2            | [3, 4, 6, 3]  | [64, 128, 320, 512] | 768                     | 25.4           | 81.6                  |
+-------------------+---------------+---------------------+-------------------------+----------------+-----------------------+
| MiT-b3            | [3, 4, 18, 3] | [64, 128, 320, 512] | 768                     | 45.2           | 83.1                  |
+-------------------+---------------+---------------------+-------------------------+----------------+-----------------------+
| MiT-b4            | [3, 8, 27, 3] | [64, 128, 320, 512] | 768                     | 62.6           | 83.6                  |
+-------------------+---------------+---------------------+-------------------------+----------------+-----------------------+
| MiT-b5            | [3, 6, 40, 3] | [64, 128, 320, 512] | 768                     | 82.0           | 83.8                  |
+-------------------+---------------+---------------------+-------------------------+----------------+-----------------------+

SegformerConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SegformerConfig
    :members:


SegformerFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SegformerFeatureExtractor
    :members: __call__


SegformerModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SegformerModel
    :members: forward


SegformerDecodeHead
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SegformerDecodeHead
    :members: forward


SegformerForImageClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SegformerForImageClassification
    :members: forward


SegformerForSemanticSegmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SegformerForSemanticSegmentation
    :members: forward
