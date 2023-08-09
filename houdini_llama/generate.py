import os

from model import ExLlama, ExLlamaCache, ExLlamaConfig
class ExLlamaModel:
    def __init__(self, model_directory):
        self.model_directory = model_directory
        self.model_config_path = os.path.join(model_directory, "config.json")
        self.model_tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        self.config = ExLlamaConfig(str(self.model_config_path))
        self.model_path = self.locate_model()

    def locate_model(self):
        for ext in [".safetensors"]:
            found = list(self.model_directory.glob(*"{0}".format(ext)))
            if found:
                return found[0]