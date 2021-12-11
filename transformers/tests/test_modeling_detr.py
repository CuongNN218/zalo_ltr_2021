# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch DETR model. """


import inspect
import math
import unittest

from transformers import DetrConfig, is_timm_available, is_vision_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_timm, require_vision, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_generation_utils import GenerationTesterMixin
from .test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor


if is_timm_available():
    import torch

    from transformers import DetrForObjectDetection, DetrForSegmentation, DetrModel


if is_vision_available():
    from PIL import Image

    from transformers import DetrFeatureExtractor


class DetrModelTester:
    def __init__(
        self,
        parent,
        batch_size=8,
        is_training=True,
        use_labels=True,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_queries=12,
        num_channels=3,
        min_size=200,
        max_size=200,
        n_targets=8,
        num_labels=91,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.min_size = min_size
        self.max_size = max_size
        self.n_targets = n_targets
        self.num_labels = num_labels

        # we also set the expected seq length for both encoder and decoder
        self.encoder_seq_length = math.ceil(self.min_size / 32) * math.ceil(self.max_size / 32)
        self.decoder_seq_length = self.num_queries

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.min_size, self.max_size])

        pixel_mask = torch.ones([self.batch_size, self.min_size, self.max_size], device=torch_device)

        labels = None
        if self.use_labels:
            # labels is a list of Dict (each Dict being the labels for a given example in the batch)
            labels = []
            for i in range(self.batch_size):
                target = {}
                target["class_labels"] = torch.randint(
                    high=self.num_labels, size=(self.n_targets,), device=torch_device
                )
                target["boxes"] = torch.rand(self.n_targets, 4, device=torch_device)
                target["masks"] = torch.rand(self.n_targets, self.min_size, self.max_size, device=torch_device)
                labels.append(target)

        config = self.get_config()
        return config, pixel_values, pixel_mask, labels

    def get_config(self):
        return DetrConfig(
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            num_queries=self.num_queries,
            num_labels=self.num_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, labels = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        return config, inputs_dict

    def create_and_check_detr_model(self, config, pixel_values, pixel_mask, labels):
        model = DetrModel(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.decoder_seq_length, self.hidden_size)
        )

    def create_and_check_detr_object_detection_head_model(self, config, pixel_values, pixel_mask, labels):
        model = DetrForObjectDetection(config=config)
        model.to(torch_device)
        model.eval()

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(pixel_values)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_queries, self.num_labels + 1))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))

        result = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_queries, self.num_labels + 1))
        self.parent.assertEqual(result.pred_boxes.shape, (self.batch_size, self.num_queries, 4))


@require_timm
class DetrModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            DetrModel,
            DetrForObjectDetection,
            DetrForSegmentation,
        )
        if is_timm_available()
        else ()
    )
    is_encoder_decoder = True
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False

    # special case for head models
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ in ["DetrForObjectDetection", "DetrForSegmentation"]:
                labels = []
                for i in range(self.model_tester.batch_size):
                    target = {}
                    target["class_labels"] = torch.ones(
                        size=(self.model_tester.n_targets,), device=torch_device, dtype=torch.long
                    )
                    target["boxes"] = torch.ones(
                        self.model_tester.n_targets, 4, device=torch_device, dtype=torch.float
                    )
                    target["masks"] = torch.ones(
                        self.model_tester.n_targets,
                        self.model_tester.min_size,
                        self.model_tester.max_size,
                        device=torch_device,
                        dtype=torch.float,
                    )
                    labels.append(target)
                inputs_dict["labels"] = labels

        return inputs_dict

    def setUp(self):
        self.model_tester = DetrModelTester(self)
        self.config_tester = ConfigTester(self, config_class=DetrConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_detr_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_detr_model(*config_and_inputs)

    def test_detr_object_detection_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_detr_object_detection_head_model(*config_and_inputs)

    @unittest.skip(reason="DETR does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="DETR does not have a get_input_embeddings method")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="DETR is not a generative model")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="DETR does not use token embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        decoder_seq_length = self.model_tester.decoder_seq_length
        encoder_seq_length = self.model_tester.encoder_seq_length
        decoder_key_length = self.model_tester.decoder_seq_length
        encoder_key_length = self.model_tester.encoder_seq_length

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )
            out_len = len(outputs)

            if self.is_encoder_decoder:
                correct_outlen = 5

                # loss is at first position
                if "labels" in inputs_dict:
                    correct_outlen += 1  # loss is added to beginning
                # Object Detection model returns pred_logits and pred_boxes
                if model_class.__name__ == "DetrForObjectDetection":
                    correct_outlen += 2
                # Panoptic Segmentation model returns pred_logits, pred_boxes, pred_masks
                if model_class.__name__ == "DetrForSegmentation":
                    correct_outlen += 3
                if "past_key_values" in outputs:
                    correct_outlen += 1  # past_key_values have been returned

                self.assertEqual(out_len, correct_outlen)

                # decoder attentions
                decoder_attentions = outputs.decoder_attentions
                self.assertIsInstance(decoder_attentions, (list, tuple))
                self.assertEqual(len(decoder_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(decoder_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, decoder_seq_length, decoder_key_length],
                )

                # cross attentions
                cross_attentions = outputs.cross_attentions
                self.assertIsInstance(cross_attentions, (list, tuple))
                self.assertEqual(len(cross_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(cross_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        decoder_seq_length,
                        encoder_key_length,
                    ],
                )

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

    def test_retain_grad_hidden_states_attentions(self):
        # removed retain_grad and grad on decoder_hidden_states, as queries don't require grad

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        encoder_hidden_states = outputs.encoder_hidden_states[0]
        encoder_attentions = outputs.encoder_attentions[0]
        encoder_hidden_states.retain_grad()
        encoder_attentions.retain_grad()

        decoder_attentions = outputs.decoder_attentions[0]
        decoder_attentions.retain_grad()

        cross_attentions = outputs.cross_attentions[0]
        cross_attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(encoder_hidden_states.grad)
        self.assertIsNotNone(encoder_attentions.grad)
        self.assertIsNotNone(decoder_attentions.grad)
        self.assertIsNotNone(cross_attentions.grad)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            if model.config.is_encoder_decoder:
                expected_arg_names = ["pixel_values", "pixel_mask"]
                expected_arg_names.extend(
                    ["head_mask", "decoder_head_mask", "encoder_outputs"]
                    if "head_mask" and "decoder_head_mask" in arg_names
                    else []
                )
                self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)
            else:
                expected_arg_names = ["pixel_values", "pixel_mask"]
                self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_different_timm_backbone(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # let's pick a random timm backbone
        config.backbone = "tf_mobilenetv3_small_075"

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if model_class.__name__ == "DetrForObjectDetection":
                expected_shape = (
                    self.model_tester.batch_size,
                    self.model_tester.num_queries,
                    self.model_tester.num_labels + 1,
                )
                self.assertEqual(outputs.logits.shape, expected_shape)

            self.assertTrue(outputs)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        configs_no_init.init_xavier_std = 1e9

        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "bbox_attention" in name and "bias" not in name:
                        self.assertLess(
                            100000,
                            abs(param.data.max().item()),
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )


TOLERANCE = 1e-4


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_timm
@require_vision
@slow
class DetrModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50") if is_vision_available() else None

    def test_inference_no_head(self):
        model = DetrModel.from_pretrained("facebook/detr-resnet-50").to(torch_device)

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        encoding = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        with torch.no_grad():
            outputs = model(**encoding)

        expected_shape = torch.Size((1, 100, 256))
        assert outputs.last_hidden_state.shape == expected_shape
        expected_slice = torch.tensor(
            [[0.0616, -0.5146, -0.4032], [-0.7629, -0.4934, -1.7153], [-0.4768, -0.6403, -0.7826]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))

    def test_inference_object_detection_head(self):
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(torch_device)

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        encoding = feature_extractor(images=image, return_tensors="pt").to(torch_device)
        pixel_values = encoding["pixel_values"].to(torch_device)
        pixel_mask = encoding["pixel_mask"].to(torch_device)

        with torch.no_grad():
            outputs = model(pixel_values, pixel_mask)

        expected_shape_logits = torch.Size((1, model.config.num_queries, model.config.num_labels + 1))
        self.assertEqual(outputs.logits.shape, expected_shape_logits)
        expected_slice_logits = torch.tensor(
            [[-19.1194, -0.0893, -11.0154], [-17.3640, -1.8035, -14.0219], [-20.0461, -0.5837, -11.1060]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits, atol=1e-4))

        expected_shape_boxes = torch.Size((1, model.config.num_queries, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        expected_slice_boxes = torch.tensor(
            [[0.4433, 0.5302, 0.8853], [0.5494, 0.2517, 0.0529], [0.4998, 0.5360, 0.9956]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4))

    def test_inference_panoptic_segmentation_head(self):
        model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic").to(torch_device)

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        encoding = feature_extractor(images=image, return_tensors="pt").to(torch_device)
        pixel_values = encoding["pixel_values"].to(torch_device)
        pixel_mask = encoding["pixel_mask"].to(torch_device)

        with torch.no_grad():
            outputs = model(pixel_values, pixel_mask)

        expected_shape_logits = torch.Size((1, model.config.num_queries, model.config.num_labels + 1))
        self.assertEqual(outputs.logits.shape, expected_shape_logits)
        expected_slice_logits = torch.tensor(
            [[-18.1565, -1.7568, -13.5029], [-16.8888, -1.4138, -14.1028], [-17.5709, -2.5080, -11.8654]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits, atol=1e-4))

        expected_shape_boxes = torch.Size((1, model.config.num_queries, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        expected_slice_boxes = torch.tensor(
            [[0.5344, 0.1789, 0.9285], [0.4420, 0.0572, 0.0875], [0.6630, 0.6887, 0.1017]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4))

        expected_shape_masks = torch.Size((1, model.config.num_queries, 200, 267))
        self.assertEqual(outputs.pred_masks.shape, expected_shape_masks)
        expected_slice_masks = torch.tensor(
            [[-7.7558, -10.8788, -11.9797], [-11.8881, -16.4329, -17.7451], [-14.7316, -19.7383, -20.3004]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.pred_masks[0, 0, :3, :3], expected_slice_masks, atol=1e-3))
