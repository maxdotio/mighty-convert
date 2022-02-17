from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np

from transformers import PreTrainedModel, PreTrainedTokenizer, TensorType, TFPreTrainedModel
from transformers.onnx.config import OnnxConfig
from transformers.utils import logging

from optimum.onnxruntime import ORTConfig, ORTOptimizer, ORTQuantizer


# Copyright 2021 The HuggingFace Team. All rights reserved.
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
def validate_model_outputs(
	config: OnnxConfig,
	tokenizer: PreTrainedTokenizer,
	reference_model: Union[PreTrainedModel, TFPreTrainedModel],
	onnx_model: Path,
	onnx_named_outputs: List[str],
	atol: float,
	kind: str,
):
	from onnxruntime import InferenceSession, SessionOptions

	logging.set_verbosity(logging.INFO)
	logger = logging.get_logger()  # pylint: disable=invalid-name
	logger.info("Validating ONNX model...")

	# TODO: generate inputs with a different batch_size and seq_len that was used for conversion to properly test
	# dynamic input shapes.
	reference_model_inputs = config.generate_dummy_inputs(tokenizer, framework=TensorType.PYTORCH)

	# Create ONNX Runtime session
	options = SessionOptions()
	session = InferenceSession(onnx_model.as_posix(), options)

	# Compute outputs from the reference model
	ref_outputs = reference_model(**reference_model_inputs)
	ref_outputs_dict = {}

	# We flatten potential collection of outputs (i.e. past_keys) to a flat structure
	for name, value in ref_outputs.items():
		# Overwriting the output name as "present" since it is the name used for the ONNX outputs
		# ("past_key_values" being taken for the ONNX inputs)
		if name == "past_key_values":
			name = "present"
		if isinstance(value, (list, tuple)):
			value = config.flatten_output_collection_property(name, value)
			ref_outputs_dict.update(value)
		else:
			ref_outputs_dict[name] = value

	# We flatten potential collection of inputs (i.e. past_keys)
	onnx_inputs = {}
	for name, value in reference_model_inputs.items():
		if isinstance(value, (list, tuple)):
			value = config.flatten_output_collection_property(name, value)
			onnx_inputs.update({tensor_name: pt_tensor.numpy() for tensor_name, pt_tensor in value.items()})
		else:
			onnx_inputs[name] = value.numpy()

	# Compute outputs from the ONNX model
	onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)

	# Check we have a subset of the keys into onnx_outputs against ref_outputs
	ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_named_outputs)
	if not onnx_outputs_set.issubset(ref_outputs_set):
		logger.info(f"\t-[x] ONNX model output names {onnx_outputs_set} do not match reference model {ref_outputs_set}")
		logger.warning(f"{kind} model generation failed.")
		raise ValueError(
			"Outputs doesn't match between reference model and ONNX exported model: "
			f"{onnx_outputs_set.difference(ref_outputs_set)}"
		)
	else:
		logger.info(f"\t-[✓] ONNX model output names match reference model ({onnx_outputs_set})")

	# Check the shape and values match
	for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
		ref_value = ref_outputs_dict[name].detach().numpy()
		logger.info(f'\t- Validating ONNX Model output "{name}":')

		# Shape
		if not ort_value.shape == ref_value.shape:
			logger.info(f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")
			logger.warning(f"{kind} model generation failed.")
			raise ValueError(
				"Outputs shape doesn't match between reference model and ONNX exported model: "
				f"Got {ref_value.shape} (reference) and {ort_value.shape} (ONNX)"
			)
		else:
			logger.info(f"\t\t-[✓] {ort_value.shape} matches {ref_value.shape}")

		# Values
		if not np.allclose(ref_value, ort_value, atol=atol):
			logger.info(f"\t\t-[x] values not close enough (atol: {atol})")
			logger.warning(f"{kind} model generation failed.")
			if kind == "Quantized":
				logger.warning(f"That's OK though - you can still use the Optimized model.")
			raise ValueError(
				"Output values do not match between the reference model and the ONNX exported model: "
				f"Got max absolute difference of: {np.amax(np.abs(ref_value - ort_value))}"
			)
		else:
			logger.info(f"\t\t-[✓] all values close (atol: {atol})")
			logger.info(f"\t\t-[✓] max absolute difference of: {np.amax(np.abs(ref_value - ort_value))}")
			logger.info(f"\t\t-[✓] avg absolute difference of: {np.mean(np.abs(ref_value - ort_value))}")
			logger.info(f"{kind} model created successfully.")

	logging.set_verbosity(logging.ERROR)
