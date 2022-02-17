#See LICENSE.md and ThirdPartyNotices.md for licensing information

import os
import json
import requests
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from transformers.onnx.config import OnnxConfig
from transformers.utils import logging

repository_dir = Path(str(os.path.abspath(os.path.join(__file__, ".." , "output"))))

def download_config(model_name):
	source = f"https://huggingface.co/{model_name}"
	url = f"{source}/raw/main/config.json"
	resp = requests.get(url=url)
	data = resp.json()
	data["name"]=model_name
	data["source"]=source
	output_dir = repository_dir.joinpath(model_name)
	config_path = output_dir.joinpath("config.json")
	with open(config_path,"w") as fd:
		json.dump(data,fd,indent=2)
	return data

def download(model_name):

	output_dir = repository_dir.joinpath(model_name)

	try:
		#get the model if there is one
		model = AutoModel.from_pretrained(model_name)
	except:
		print(f"Oops!  Couldn't download the model {model_name}")

	try:
		#get the tokenizer if there is one
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		tokenizer.save_pretrained(save_directory=output_dir,legacy_format=False,push_to_hub=False)
	except:
		print(f"Oops!  Couldn't download the tokenizer {model_name}")

	try:
		#get the config if there is one
		#config = AutoConfig.from_pretrained(model_name)
		config = download_config(model_name)
		print(config)
	except:
		print(f"Oops!  Couldn't download the model configuration {model_name}")
