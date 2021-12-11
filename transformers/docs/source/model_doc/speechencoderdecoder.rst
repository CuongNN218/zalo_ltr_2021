.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Speech Encoder Decoder Models
-----------------------------------------------------------------------------------------------------------------------

The :class:`~transformers.SpeechEncoderDecoderModel` can be used to initialize a speech-sequence-to-text-sequence model
with any pretrained speech autoencoding model as the encoder (*e.g.* :doc:`Wav2Vec2 <wav2vec2>`, :doc:`Hubert
<hubert>`) and any pretrained autoregressive model as the decoder.

The effectiveness of initializing speech-sequence-to-text-sequence models with pretrained checkpoints for speech
recognition and speech translation has *e.g.* been shown in `Large-Scale Self- and Semi-Supervised Learning for Speech
Translation <https://arxiv.org/abs/2104.06678>`__ by Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli,
Alexis Conneau.

An example of how to use a :class:`~transformers.SpeechEncoderDecoderModel` for inference can be seen in
:doc:`Speech2Text2 <speech_to_text_2>`.


SpeechEncoderDecoderConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SpeechEncoderDecoderConfig
    :members:


SpeechEncoderDecoderModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.SpeechEncoderDecoderModel
    :members: forward, from_encoder_decoder_pretrained
