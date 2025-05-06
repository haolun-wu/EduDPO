""" Wrapper class around common HuggingFace model loading and inference functionalities """

import os 
import torch 
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    BitsAndBytesConfig, pipeline
)
from peft import AutoPeftModelForCausalLM
from utils.cuda import supports_flash_attention


class HuggingFaceLocalModel():
    
    def __init__(self, config, is_adapter=False, is_training=False) -> None:
        """
        Initialize the HuggingFaceLocalModel model.

        Args:
            config (dict): The configuration of the model.
            is_adapter (bool, optional): Whether to use an adapter model. Defaults to False.
            is_training (bool, optional): Whether to use the model for training or not. Defaults to False.
        """

        self.config = config 
        self.is_adapter = is_adapter
        self.is_training = is_training

        self.supports_flash_attention = supports_flash_attention()
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()           

    def load_model(self):
        """
        Load a model from the config parameters.

        The model is loaded using the transformers library AutoModelForCausalLM
        or AutoPeftModelForCausalLM if is_adapter is True. The model is loaded
        with the specified dtype and device map.

        If the model is a quantized model, it is loaded with the specified
        quantization configuration.

        If the model is an adapter model, it is loaded with the specified
        adapter configuration.

        The model is then converted to the specified dtype and device map.

        Returns:
            The loaded model.
        """

        if self.config.dtype == "fp32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16

        if self.supports_flash_attention:
            attn_implementation = "flash_attention_2"
            torch_dtype = torch.bfloat16
        else:
            attn_implementation = "eager"

        if self.config.quant == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.quant == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None 
        
        
        device_map = get_kbit_device_map()
        if self.config.device_map: 
            device_map = self.config.device_map

        if self.is_adapter:
            model = AutoPeftModelForCausalLM.from_pretrained(
                self.config.name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
                is_trainable=self.is_training
            )

            if self.is_training:
                model.print_trainable_parameters()
            else:
                if bnb_config is None:
                    model = model.merge_and_unload()
                    print("merged model")

        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.name,
                quantization_config=bnb_config,
                torch_dtype=torch_dtype,
                device_map = device_map, 
                attn_implementation=attn_implementation,
            )

        print("dtype", torch_dtype)
        print("attention", attn_implementation)
        print("device map", device_map)
        print("Model out of LocalAgent", model)
        print("huggingface device map", model.hf_device_map)

        return model 


    def load_tokenizer(self):
        """
        Load the tokenizer from the pretrained model specified in the configuration.

        The tokenizer is configured to ensure compatibility with the model's 
        requirements. If the pad token is not set, it defaults to the end-of-sequence 
        token. The padding and truncation are set to 'left' to handle specific 
        generation conditions.

        Returns:
            The configured tokenizer instance.
        """

        tokenizer = AutoTokenizer.from_pretrained(self.config.name)
        if tokenizer.pad_token == None: 
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left' # to prevent errors with FA
        tokenizer.truncation_side = 'left' # to prevent cutting off last generation

        return tokenizer 
    

    def load_pipeline(self):
        return pipeline("text-generation", 
                        model=self.model, 
                        tokenizer=self.tokenizer)
    

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map():
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None

