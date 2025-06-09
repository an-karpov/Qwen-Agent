import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

PHI3MODEL_PATH = ""
LLAMA_PATH = ""

class HFModel(object):

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          trust_remote_code=True,
                                                          device_map='auto',
                                                          low_cpu_mem_usage=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model.generation_config.do_sample = False

class LLM(HFModel):

    def __init__(self, model_path):
        super().__init__(model_path)

    def generate(self, input_text, stop_words=[], max_new_tokens=512):
        if isinstance(input_text, str):
            input_text = [input_text]

        input_ids = self.tokenizer(input_text)['input_ids']
        input_ids = torch.tensor(input_ids, device=self.model.device)
        gen_kwargs = {'max_new_tokens': max_new_tokens, 'do_sample': False}
        outputs = self.model.generate(input_ids, **gen_kwargs)
        s = outputs[0][input_ids.shape[1]:]
        output = self.tokenizer.decode(s, skip_special_tokens=True)

        for stop_str in stop_words:
            idx = output.find(stop_str)
            if idx != -1:
                output = output[:idx + len(stop_str)]

        return output

class Phi3Model(HFModel):
    def __init__(self):
        super().__init__(PHI3MODEL_PATH)
    