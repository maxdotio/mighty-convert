# mighty-convert

Converts a Transformer model, tokenizer, and config to be compatible with Mighty Inference Server (https://max.io/)

Just provide a model name and a pipeline!

```
./mighty-convert [MODEL_NAME] [PIPELINE]
```

## Examples:

### Embeddings:
```
./mighty-convert microsoft/xtremedistil-l12-h384-uncased default
```

### Sequence Classification/Sentiment Analysis
```
./mighty-convert cardiffnlp/twitter-roberta-base-sentiment sequence-classification

```

### Sentence Transformers
```
./mighty-convert sentence-transformers/all-MiniLM-L6-v2 sentence-transformers
```

### Question Answering
```
./mighty-convert deepset/roberta-base-squad2 question-answering
```

# Install

Requires Python 3.8+

```
git clone https://github.com/binarymax/mighty-convert
cd mighty-convert
pip install -r requirements.txt
```

# Run

```
./mighty-convert [MODEL_NAME] [PIPELINE]
```

The above command will output all files needed to use the model with Mighty Inference Server.

This includes:

- config.json
- tokenizer.json
- model-optimized.onnx
- model-quantized.onnx _(if accuracy is within reasonable tolerance)_


# Compatibility:

Currently supported pipelines:

- default
- sentence-transformers
- sequence-classification
- question-answering

The model and pipeline must be compatible with each other!  For example, a model trained on sentence-transformers cannot be converted to question-answering.

For the model name, only Huggingface.co URL paths are supported (like `microsoft/xtremedistil-l12-h384-uncased`).

ONNX is the model conversion format output, and has a maximum size of 2GB.  To check the model size look for the largest '.bin' file in the source Huggingface model's files.
