# HuggingFace Auto-Quantizer

This automatically quantizes any causal language model with weights hosted on HuggingFace to q8.

Any model is compatible if it can be loaded with:
```python3
tokenizer = AutoTokenizer.from_pretrained(model_name, ...)
model = AutoModelForCausalLM.from_pretrained(model_name, ...)
```

## How it works

This is a clever use of the [new `train` functionality in Cog 0.9+](https://cog.run/training).

It takes in `model_name`, and then runs `train.py:train` which loads the tokenizer & model in **q8**.
It then saves the weights to disk, tarballs and compresses them, and returns the result path.

On Replicate, this saves the output weights to a new model.

