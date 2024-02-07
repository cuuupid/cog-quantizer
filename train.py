from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from cog import BaseModel, Input, Path
import tarfile
import torch
import os.path

def train(
    model_name: str = Input(description="Model's name from HuggingFace (username/name format)"),
) -> Path:
    user, name = model_name.split('/')
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        use_cache=True,
        load_in_8bit=True # q8
        # llm_int8_enable_fp32_cpu_offload=True
    )
    print("Saving quantized model and tokenizer...")
    model.save_pretrained('./quantized/model')
    tokenizer.save_pretrained('./quantized/tokenizer')
    print("Creating tar file...")
    with tarfile.open(f"{name}.tar", "w:gz") as tar:
        tar.add('./quantized', arcname=os.path.sep)
    return Path(f"{name}.tar")
