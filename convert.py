#See LICENSE.md and ThirdPartyNotices.md for licensing information

import os
import gc
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Union
from packaging.version import Version, parse
from optimum.onnxruntime import ORTConfig, ORTOptimizer, ORTQuantizer

#Local packages
import download
import validate

feature_map = {
	"default":"default",
	"sequence-classification":"sequence-classification",
	"question-answering":"question-answering",
	"sentence-transformers":"default",
	"token-classification":"token-classification",
	#"masked-lm":"masked-lm",
	#"causal-lm":"causal-lm",
	#"seq2seq-lm":"seq2seq-lm",
	#"multiple-choice":"multiple-choice",
}

repository_dir = str(os.path.abspath(os.path.join(__file__, ".." , "output")))

def optimization(model_name,feature):
	ort_config = ORTConfig()
	ort_config.only_onnxruntime = False
	output_dir = Path(str(os.path.abspath(os.path.join(repository_dir,model_name))))
	optim_model_path = output_dir.joinpath("model-optimized.onnx")
	optimizer = ORTOptimizer(ort_config)
	optimizer.fit(model_name, Path(output_dir), feature=feature)
	optimizer.get_optimize_details()
	validate.validate_model_outputs(
		optimizer.onnx_config,
		optimizer.tokenizer,
		optimizer.model,
		optim_model_path,
		list(optimizer.onnx_config.outputs.keys()),
		atol=1e-4,
		kind="Optimized"
	)
	gc.collect()

def dynamic_quantization(model_name,feature):
	ort_config = ORTConfig()
	ort_config.only_onnxruntime = False
	ort_config.quantization_approach = "dynamic"
	ort_config.optimize_model = True
	output_dir = Path(str(os.path.abspath(os.path.join(repository_dir,model_name))))
	q8_model_path = output_dir.joinpath("model-quantized.onnx")
	quantizer = ORTQuantizer(ort_config)
	quantizer.fit(model_name, Path(output_dir), feature=feature)
	validate.validate_model_outputs(
		quantizer.onnx_config,
		quantizer.tokenizer,
		quantizer.model,
		q8_model_path,
		list(quantizer.onnx_config.outputs.keys()),
		atol=8e-1,
		kind="Quantized"
	)
	gc.collect()

if __name__ == "__main__":
	
	model_name = sys.argv[1]
	feature = sys.argv[2]

	if feature in feature_map:

		#Map the common name to the pipeline name
		feature = feature_map[feature]
	
		if True:
			print("---------------------------------------")
			print(f"Downloading {model_name} for task {feature}")
			download.download(model_name)

		if True:
			print("---------------------------------------")
			print(f"Optimizing {model_name}")
			optimization(model_name,feature)

		if True:
			print("---------------------------------------")
			print(f"Quantizing {model_name}")
			dynamic_quantization(model_name,feature)
	
	else:
		print(f"Sorry, '{feature}'' is not a recognized pipeline!")
		print(f"Only models with the following possible pipelines are supported: {[k for k in feature_map.keys()]}")