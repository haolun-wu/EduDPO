""" Base Wrapper class around HuggingFace 
Instruction-tuned and chat models.

"""

import torch 
from transformers import GenerationConfig

# TODO: change the name to text generation pipeline 
class CustomPipeline():
    """
    
    Custom class that imitates HuggingFace Transformers pipeline for text generation.
    Useful in case where pipeline is not available for certain type
    of inference 
    """

    def __init__(self, mode, model, tokenizer, return_full_text=False) -> None:
        self.mode = mode
        self.model = model
        self.tokenizer = tokenizer
        self.return_full_text = return_full_text
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, input_messages, **gen_kwargs):
        # Adapting generation configurations
        gen_conf = GenerationConfig(**gen_kwargs)
        gen_conf.eos_token_id = self.tokenizer.eos_token_id
        gen_conf.pad_token_id = self.tokenizer.pad_token_id 

        # Generating the responses
        model_inputs = self.encode(input_messages)
        model_inputs = model_inputs.to(self.device) 
        with torch.no_grad():
            gen_outputs = self.model.generate(**model_inputs, 
                                              generation_config=gen_conf)
            
        # Only returning the generated portion 
        if not self.return_full_text:
            input_ids = model_inputs.input_ids
            n_generations = len(input_ids)
            gen_outputs = [gen_outputs[i][len(input_ids[i]):]
                           for i in range(n_generations)]
        
        outputs = self.decode(gen_outputs)
        return outputs
    

    def encode(self, inputs, **kwargs):
        return self.tokenizer(inputs, return_tensors="pt",
                              truncation=True, padding=True,
                              #pad_to_multiple_of=8,
                              **kwargs)
    

    def decode(self, output_ids):
        outputs = self.tokenizer.batch_decode(output_ids, 
                                              skip_special_tokens=True)
        return list(outputs)
    