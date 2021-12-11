.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Encoder Decoder Models
-----------------------------------------------------------------------------------------------------------------------

The :class:`~transformers.EncoderDecoderModel` can be used to initialize a sequence-to-sequence model with any
pretrained autoencoding model as the encoder and any pretrained autoregressive model as the decoder.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks
was shown in `Leveraging Pre-trained Checkpoints for Sequence Generation Tasks <https://arxiv.org/abs/1907.12461>`__ by
Sascha Rothe, Shashi Narayan, Aliaksei Severyn.

After such an :class:`~transformers.EncoderDecoderModel` has been trained/fine-tuned, it can be saved/loaded just like
any other models (see the examples for more information).

An application of this architecture could be to leverage two pretrained :class:`~transformers.BertModel` as the encoder
and decoder for a summarization model as was shown in: `Text Summarization with Pretrained Encoders
<https://arxiv.org/abs/1908.08345>`__ by Yang Liu and Mirella Lapata.

The :meth:`~transformers.TFEncoderDecoderModel.from_pretrained` currently doesn't support initializing the model from a
pytorch checkpoint. Passing ``from_pt=True`` to this method will throw an exception. If there are only pytorch
checkpoints for a particular encoder-decoder model, a workaround is:

.. code-block::

    >>> # a workaround to load from pytorch checkpoint
    >>> _model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")
    >>> _model.encoder.save_pretrained("./encoder")
    >>> _model.decoder.save_pretrained("./decoder")
    >>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
    ...     "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
    ... )
    >>> # This is only for copying some specific attributes of this particular model.
    >>> model.config = _model.config

This model was contributed by `thomwolf <https://github.com/thomwolf>`__. This model's TensorFlow and Flax versions
were contributed by `ydshieh <https://github.com/ydshieh>`__.


EncoderDecoderConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.EncoderDecoderConfig
    :members:


EncoderDecoderModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.EncoderDecoderModel
    :members: forward, from_encoder_decoder_pretrained


TFEncoderDecoderModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFEncoderDecoderModel
    :members: call, from_encoder_decoder_pretrained


FlaxEncoderDecoderModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaxEncoderDecoderModel
    :members: __call__, from_encoder_decoder_pretrained
