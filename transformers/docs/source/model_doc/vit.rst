.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Vision Transformer (ViT)
-----------------------------------------------------------------------------------------------------------------------

.. note::

    This is a recently introduced model so the API hasn't been tested extensively. There may be some bugs or slight
    breaking changes to fix it in the future. If you see something strange, file a `Github Issue
    <https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title>`__.


Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Vision Transformer (ViT) model was proposed in `An Image is Worth 16x16 Words: Transformers for Image Recognition
at Scale <https://arxiv.org/abs/2010.11929>`__ by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk
Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob
Uszkoreit, Neil Houlsby. It's the first paper that successfully trains a Transformer encoder on ImageNet, attaining
very good results compared to familiar convolutional architectures.


The abstract from the paper is the following:

*While the Transformer architecture has become the de-facto standard for natural language processing tasks, its
applications to computer vision remain limited. In vision, attention is either applied in conjunction with
convolutional networks, or used to replace certain components of convolutional networks while keeping their overall
structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to
sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of
data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.),
Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring
substantially fewer computational resources to train.*

Tips:

- To feed images to the Transformer encoder, each image is split into a sequence of fixed-size non-overlapping patches,
  which are then linearly embedded. A [CLS] token is added to serve as representation of an entire image, which can be
  used for classification. The authors also add absolute position embeddings, and feed the resulting sequence of
  vectors to a standard Transformer encoder.
- As the Vision Transformer expects each image to be of the same size (resolution), one can use
  :class:`~transformers.ViTFeatureExtractor` to resize (or rescale) and normalize images for the model.
- Both the patch resolution and image resolution used during pre-training or fine-tuning are reflected in the name of
  each checkpoint. For example, :obj:`google/vit-base-patch16-224` refers to a base-sized architecture with patch
  resolution of 16x16 and fine-tuning resolution of 224x224. All checkpoints can be found on the `hub
  <https://huggingface.co/models?search=vit>`__.
- The available checkpoints are either (1) pre-trained on `ImageNet-21k <http://www.image-net.org/>`__ (a collection of
  14 million images and 21k classes) only, or (2) also fine-tuned on `ImageNet
  <http://www.image-net.org/challenges/LSVRC/2012/>`__ (also referred to as ILSVRC 2012, a collection of 1.3 million
  images and 1,000 classes).
- The Vision Transformer was pre-trained using a resolution of 224x224. During fine-tuning, it is often beneficial to
  use a higher resolution than pre-training `(Touvron et al., 2019) <https://arxiv.org/abs/1906.06423>`__, `(Kolesnikov
  et al., 2020) <https://arxiv.org/abs/1912.11370>`__. In order to fine-tune at higher resolution, the authors perform
  2D interpolation of the pre-trained position embeddings, according to their location in the original image.
- The best results are obtained with supervised pre-training, which is not the case in NLP. The authors also performed
  an experiment with a self-supervised pre-training objective, namely masked patched prediction (inspired by masked
  language modeling). With this approach, the smaller ViT-B/16 model achieves 79.9% accuracy on ImageNet, a significant
  improvement of 2% to training from scratch, but still 4% behind supervised pre-training.

Following the original Vision Transformer, some follow-up works have been made:

- DeiT (Data-efficient Image Transformers) by Facebook AI. DeiT models are distilled vision transformers. Refer to
  :doc:`DeiT's documentation page <deit>`. The authors of DeiT also released more efficiently trained ViT models, which
  you can directly plug into :class:`~transformers.ViTModel` or :class:`~transformers.ViTForImageClassification`. There
  are 4 variants available (in 3 different sizes): `facebook/deit-tiny-patch16-224`, `facebook/deit-small-patch16-224`,
  `facebook/deit-base-patch16-224` and `facebook/deit-base-patch16-384`. Note that one should use
  :class:`~transformers.DeiTFeatureExtractor` in order to prepare images for the model.

- BEiT (BERT pre-training of Image Transformers) by Microsoft Research. BEiT models outperform supervised pre-trained
  vision transformers using a self-supervised method inspired by BERT (masked image modeling) and based on a VQ-VAE.
  Refer to :doc:`BEiT's documentation page <beit>`.

- DINO (a method for self-supervised training of Vision Transformers) by Facebook AI. Vision Transformers trained using
  the DINO method show very interesting properties not seen with convolutional models. They are capable of segmenting
  objects, without having ever been trained to do so. DINO checkpoints can be found on the `hub
  <https://huggingface.co/models?other=dino>`__.

This model was contributed by `nielsr <https://huggingface.co/nielsr>`__. The original code (written in JAX) can be
found `here <https://github.com/google-research/vision_transformer>`__.

Note that we converted the weights from Ross Wightman's `timm library
<https://github.com/rwightman/pytorch-image-models>`__, who already converted the weights from JAX to PyTorch. Credits
go to him!


ViTConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ViTConfig
    :members:


ViTFeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ViTFeatureExtractor
    :members: __call__


ViTModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ViTModel
    :members: forward


ViTForImageClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ViTForImageClassification
    :members: forward


TFViTModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFViTModel
    :members: call


TFViTForImageClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFViTForImageClassification
    :members: call


FlaxVitModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaxViTModel
    :members: __call__


FlaxViTForImageClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaxViTForImageClassification
    :members: __call__

