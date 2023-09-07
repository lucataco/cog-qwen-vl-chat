# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MODEL_NAME = "Qwen/Qwen-VL-Chat"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=TOKEN_CACHE
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE
        )
        model.generation_config = GenerationConfig.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE
        )
        self.model = model.to("cuda")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Question", default="What is the name of the movie in the poster?"),
    ) -> str:
        """Run a single prediction on the model"""
        query = self.tokenizer.from_list_format([
            {'image': str(image)},
            {'text': prompt},
        ])

        response, history = self.model.chat(tokenizer=self.tokenizer, query=query, history=None)
        return response
