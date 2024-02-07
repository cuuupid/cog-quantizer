from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from cog import BasePredictor, Input, Path
import tarfile
import os.path

class Predictor(BasePredictor):
    def setup(self) -> None:
        return None

    def predict(
        self,
        model_name: str = Input(description="Model's name from HuggingFace (username/name format)"),
    ) -> Path:
        user, name = model_name.split('/')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            use_cache=True,
            load_in_8bit=True, # q8
            llm_int8_enable_fp32_cpu_offload=True
        )
        model.save_pretrained('./quantized/model')
        tokenizer.save_pretrained('./quantized/tokenizer')
        with tarfile.open(f"{name}.tar", "w:gz") as tar:
            tar.add('./quantized', arcname=os.path.sep)
        return Path(f"{name}.tar")
