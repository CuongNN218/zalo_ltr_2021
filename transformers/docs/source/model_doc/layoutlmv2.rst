.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

LayoutLMV2
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LayoutLMV2 model was proposed in `LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding
<https://arxiv.org/abs/2012.14740>`__ by Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu,
Dinei Florencio, Cha Zhang, Wanxiang Che, Min Zhang, Lidong Zhou. LayoutLMV2 improves `LayoutLM <layoutlm>`__ to obtain
state-of-the-art results across several document image understanding benchmarks:

- information extraction from scanned documents: the `FUNSD <https://guillaumejaume.github.io/FUNSD/>`__ dataset (a
  collection of 199 annotated forms comprising more than 30,000 words), the `CORD <https://github.com/clovaai/cord>`__
  dataset (a collection of 800 receipts for training, 100 for validation and 100 for testing), the `SROIE
  <https://rrc.cvc.uab.es/?ch=13>`__ dataset (a collection of 626 receipts for training and 347 receipts for testing)
  and the `Kleister-NDA <https://github.com/applicaai/kleister-nda>`__ dataset (a collection of non-disclosure
  agreements from the EDGAR database, including 254 documents for training, 83 documents for validation, and 203
  documents for testing).
- document image classification: the `RVL-CDIP <https://www.cs.cmu.edu/~aharley/rvl-cdip/>`__ dataset (a collection of
  400,000 images belonging to one of 16 classes).
- document visual question answering: the `DocVQA <https://arxiv.org/abs/2007.00398>`__ dataset (a collection of 50,000
  questions defined on 12,000+ document images).

The abstract from the paper is the following:

*Pre-training of text and layout has proved effective in a variety of visually-rich document understanding tasks due to
its effective model architecture and the advantage of large-scale unlabeled scanned/digital-born documents. In this
paper, we present LayoutLMv2 by pre-training text, layout and image in a multi-modal framework, where new model
architectures and pre-training tasks are leveraged. Specifically, LayoutLMv2 not only uses the existing masked
visual-language modeling task but also the new text-image alignment and text-image matching tasks in the pre-training
stage, where cross-modality interaction is better learned. Meanwhile, it also integrates a spatial-aware self-attention
mechanism into the Transformer architecture, so that the model can fully understand the relative positional
relationship among different text blocks. Experiment results show that LayoutLMv2 outperforms strong baselines and
achieves new state-of-the-art results on a wide variety of downstream visually-rich document understanding tasks,
including FUNSD (0.7895 -> 0.8420), CORD (0.9493 -> 0.9601), SROIE (0.9524 -> 0.9781), Kleister-NDA (0.834 -> 0.852),
RVL-CDIP (0.9443 -> 0.9564), and DocVQA (0.7295 -> 0.8672). The pre-trained LayoutLMv2 model is publicly available at
this https URL.*

Tips:

- The main difference between LayoutLMv1 and LayoutLMv2 is that the latter incorporates visual embeddings during
  pre-training (while LayoutLMv1 only adds visual embeddings during fine-tuning).
- LayoutLMv2 adds both a relative 1D attention bias as well as a spatial 2D attention bias to the attention scores in
  the self-attention layers. Details can be found on page 5 of the `paper <https://arxiv.org/abs/2012.14740>`__.
- Demo notebooks on how to use the LayoutLMv2 model on RVL-CDIP, FUNSD, DocVQA, CORD can be found `here
  <https://github.com/NielsRogge/Transformers-Tutorials>`__.
- LayoutLMv2 uses Facebook AI's `Detectron2 <https://github.com/facebookresearch/detectron2/>`__ package for its visual
  backbone. See `this link <https://detectron2.readthedocs.io/en/latest/tutorials/install.html>`__ for installation
  instructions.
- In addition to :obj:`input_ids`, :meth:`~transformer.LayoutLMv2Model.forward` expects 2 additional inputs, namely
  :obj:`image` and :obj:`bbox`. The :obj:`image` input corresponds to the original document image in which the text
  tokens occur. The model expects each document image to be of size 224x224. This means that if you have a batch of
  document images, :obj:`image` should be a tensor of shape (batch_size, 3, 224, 224). This can be either a
  :obj:`torch.Tensor` or a :obj:`Detectron2.structures.ImageList`. You don't need to normalize the channels, as this is
  done by the model. Important to note is that the visual backbone expects BGR channels instead of RGB, as all models
  in Detectron2 are pre-trained using the BGR format. The :obj:`bbox` input are the bounding boxes (i.e. 2D-positions)
  of the input text tokens. This is identical to :class:`~transformer.LayoutLMModel`. These can be obtained using an
  external OCR engine such as Google's `Tesseract <https://github.com/tesseract-ocr/tesseract>`__ (there's a `Python
  wrapper <https://pypi.org/project/pytesseract/>`__ available). Each bounding box should be in (x0, y0, x1, y1)
  format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1)
  represents the position of the lower right corner. Note that one first needs to normalize the bounding boxes to be on
  a 0-1000 scale. To normalize, you can use the following function:

.. code-block::

    def normalize_bbox(bbox, width, height):
         return [
             int(1000 * (bbox[0] / width)),
             int(1000 * (bbox[1] / height)),
             int(1000 * (bbox[2] / width)),
             int(1000 * (bbox[3] / height)),
         ]

Here, :obj:`width` and :obj:`height` correspond to the width and height of the original document in which the token
occurs (before resizing the image). Those can be obtained using the Python Image Library (PIL) library for example, as
follows:

.. code-block::

    from PIL import Image

    image = Image.open("name_of_your_document - can be a png file, pdf, etc.")

    width, height = image.size

However, this model includes a brand new :class:`~transformer.LayoutLMv2Processor` which can be used to directly
prepare data for the model (including applying OCR under the hood). More information can be found in the "Usage"
section below.

- Internally, :class:`~transformer.LayoutLMv2Model` will send the :obj:`image` input through its visual backbone to
  obtain a lower-resolution feature map, whose shape is equal to the :obj:`image_feature_pool_shape` attribute of
  :class:`~transformer.LayoutLMv2Config`. This feature map is then flattened to obtain a sequence of image tokens. As
  the size of the feature map is 7x7 by default, one obtains 49 image tokens. These are then concatenated with the text
  tokens, and send through the Transformer encoder. This means that the last hidden states of the model will have a
  length of 512 + 49 = 561, if you pad the text tokens up to the max length. More generally, the last hidden states
  will have a shape of :obj:`seq_length` + :obj:`image_feature_pool_shape[0]` *
  :obj:`config.image_feature_pool_shape[1]`.
- When calling :meth:`~transformer.LayoutLMv2Model.from_pretrained`, a warning will be printed with a long list of
  parameter names that are not initialized. This is not a problem, as these parameters are batch normalization
  statistics, which are going to have values when fine-tuning on a custom dataset.
- If you want to train the model in a distributed environment, make sure to call :meth:`synchronize_batch_norm` on the
  model in order to properly synchronize the batch normalization layers of the visual backbone.

In addition, there's LayoutXLM, which is a multilingual version of LayoutLMv2. More information can be found on
:doc:`LayoutXLM's documentation page <layoutxlm>`.

Usage: LayoutLMv2Processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to prepare data for the model is to use :class:`~transformer.LayoutLMv2Processor`, which internally
combines a feature extractor (:class:`~transformer.LayoutLMv2FeatureExtractor`) and a tokenizer
(:class:`~transformer.LayoutLMv2Tokenizer` or :class:`~transformer.LayoutLMv2TokenizerFast`). The feature extractor
handles the image modality, while the tokenizer handles the text modality. A processor combines both, which is ideal
for a multi-modal model like LayoutLMv2. Note that you can still use both separately, if you only want to handle one
modality.

.. code-block::

    from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2TokenizerFast, LayoutLMv2Processor

    feature_extractor = LayoutLMv2FeatureExtractor() # apply_ocr is set to True by default
    tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
    processor = LayoutLMv2Processor(feature_extractor, tokenizer)

In short, one can provide a document image (and possibly additional data) to :class:`~transformer.LayoutLMv2Processor`,
and it will create the inputs expected by the model. Internally, the processor first uses
:class:`~transformer.LayoutLMv2FeatureExtractor` to apply OCR on the image to get a list of words and normalized
bounding boxes, as well to resize the image to a given size in order to get the :obj:`image` input. The words and
normalized bounding boxes are then provided to :class:`~transformer.LayoutLMv2Tokenizer` or
:class:`~transformer.LayoutLMv2TokenizerFast`, which converts them to token-level :obj:`input_ids`,
:obj:`attention_mask`, :obj:`token_type_ids`, :obj:`bbox`. Optionally, one can provide word labels to the processor,
which are turned into token-level :obj:`labels`.

:class:`~transformer.LayoutLMv2Processor` uses `PyTesseract <https://pypi.org/project/pytesseract/>`__, a Python
wrapper around Google's Tesseract OCR engine, under the hood. Note that you can still use your own OCR engine of
choice, and provide the words and normalized boxes yourself. This requires initializing
:class:`~transformer.LayoutLMv2FeatureExtractor` with :obj:`apply_ocr` set to :obj:`False`.

In total, there are 5 use cases that are supported by the processor. Below, we list them all. Note that each of these
use cases work for both batched and non-batched inputs (we illustrate them for non-batched inputs).

**Use case 1: document image classification (training, inference) + token classification (inference), apply_ocr =
True**

This is the simplest case, in which the processor (actually the feature extractor) will perform OCR on the image to get
the words and normalized bounding boxes.

.. code-block::

    from transformers import LayoutLMv2Processor
    from PIL import Image

    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

    image = Image.open("name_of_your_document - can be a png file, pdf, etc.").convert("RGB")
    encoding = processor(image, return_tensors="pt") # you can also add all tokenizer parameters here such as padding, truncation
    print(encoding.keys())
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])

**Use case 2: document image classification (training, inference) + token classification (inference), apply_ocr=False**

In case one wants to do OCR themselves, one can initialize the feature extractor with :obj:`apply_ocr` set to
:obj:`False`. In that case, one should provide the words and corresponding (normalized) bounding boxes themselves to
the processor.

.. code-block::

    from transformers import LayoutLMv2Processor
    from PIL import Image

    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    image = Image.open("name_of_your_document - can be a png file, pdf, etc.").convert("RGB")
    words = ["hello", "world"]
    boxes = [[1, 2, 3, 4], [5, 6, 7, 8]] # make sure to normalize your bounding boxes
    encoding = processor(image, words, boxes=boxes, return_tensors="pt")
    print(encoding.keys())
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])

**Use case 3: token classification (training), apply_ocr=False**

For token classification tasks (such as FUNSD, CORD, SROIE, Kleister-NDA), one can also provide the corresponding word
labels in order to train a model. The processor will then convert these into token-level :obj:`labels`. By default, it
will only label the first wordpiece of a word, and label the remaining wordpieces with -100, which is the
:obj:`ignore_index` of PyTorch's CrossEntropyLoss. In case you want all wordpieces of a word to be labeled, you can
initialize the tokenizer with :obj:`only_label_first_subword` set to :obj:`False`.

.. code-block::

    from transformers import LayoutLMv2Processor
    from PIL import Image

    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    image = Image.open("name_of_your_document - can be a png file, pdf, etc.").convert("RGB")
    words = ["hello", "world"]
    boxes = [[1, 2, 3, 4], [5, 6, 7, 8]] # make sure to normalize your bounding boxes
    word_labels = [1, 2]
    encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")
    print(encoding.keys())
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'labels', 'image'])

**Use case 4: visual question answering (inference), apply_ocr=True**

For visual question answering tasks (such as DocVQA), you can provide a question to the processor. By default, the
processor will apply OCR on the image, and create [CLS] question tokens [SEP] word tokens [SEP].

.. code-block::

    from transformers import LayoutLMv2Processor
    from PIL import Image

    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

    image = Image.open("name_of_your_document - can be a png file, pdf, etc.").convert("RGB")
    question = "What's his name?"
    encoding = processor(image, question, return_tensors="pt") 
    print(encoding.keys())
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])

**Use case 5: visual question answering (inference), apply_ocr=False**

For visual question answering tasks (such as DocVQA), you can provide a question to the processor. If you want to
perform OCR yourself, you can provide your own words and (normalized) bounding boxes to the processor.

.. code-block::

    from transformers import LayoutLMv2Processor
    from PIL import Image

    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

    image = Image.open("name_of_your_document - can be a png file, pdf, etc.").convert("RGB")
    question = "What's his name?"
    words = ["hello", "world"]
    boxes = [[1, 2, 3, 4], [5, 6, 7, 8]] # make sure to normalize your bounding boxes
    encoding = processor(image, question, words, boxes=boxes, return_tensors="pt")  
    print(encoding.keys())
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'image'])

LayoutLMv2Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2Config
    :members:


LayoutLMv2FeatureExtractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2FeatureExtractor
    :members: __call__


LayoutLMv2Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2Tokenizer
    :members: __call__, save_vocabulary


LayoutLMv2TokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2TokenizerFast
    :members: __call__


LayoutLMv2Processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2Processor
    :members: __call__


LayoutLMv2Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2Model
    :members: forward


LayoutLMv2ForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2ForSequenceClassification
    :members:


LayoutLMv2ForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2ForTokenClassification
    :members:


LayoutLMv2ForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.LayoutLMv2ForQuestionAnswering
    :members:
