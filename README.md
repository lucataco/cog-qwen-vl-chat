# QwenLM/Qwen-VL-Chat Cog model

This is an implementation of [QwenLM/Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i image=@poster.jpeg -i prompt="What is the name of the movie in the poster?"
