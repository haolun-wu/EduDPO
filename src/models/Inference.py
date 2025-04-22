from copy import deepcopy 
from transformers import pipeline

from src.models.CustomPipeline import CustomPipeline

class Inference():

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.needs_custom_pipeline = False
        self.pipe = self.load_pipeline()


    def batch_query(self, batch, gen_kwargs):
        """
        Batch query the model for a list of messages.

        Args:
            batch (list of dict): A list of messages in the format of 
                [{"role": "user", "content": "message"}, ...].
            gen_kwargs (dict): Generation arguments.

        Returns:
            list of str: The generated responses.
        """

        tokenizer = self.pipe.tokenizer
        new_kwargs = adapt_gen_kwargs(deepcopy(gen_kwargs))

        agp = batch[-1][-1]["role"] == "user"

        # For trained models, this pipeline is more efficeint
        inputs = tokenizer.apply_chat_template(batch, tokenize=False, 
                                               add_generation_prompt=agp,
                                               continue_final_message=not agp,
                                               pad_to_multiple_of=8)
        responses = self.pipe(inputs, return_full_text=False, **new_kwargs)
        
        #if True or not self.needs_custom_pipeline:
        responses = [resp[j]['generated_text'] 
                     for resp in responses 
                     for j in range(len(resp))]
        
        return responses
        

    def query(self, messages, gen_kwargs):
        return self.batch_query([messages], gen_kwargs)


    def load_pipeline(self):
        self.batch_size = 2
        pipe_cls = pipeline
        if self.needs_custom_pipeline:
            pipe_cls = CustomPipeline

        return pipe_cls("text-generation", 
                        model=self.model, 
                        tokenizer=self.tokenizer)

def adapt_gen_kwargs(gen_kwargs):

    if "force_words_ids" in gen_kwargs and gen_kwargs["force_words_ids"] is None:
        gen_kwargs.pop("force_words_ids")

    gen_kwargs.pop("seed", None)

    if "n" in gen_kwargs and "num_return_sequences" not in gen_kwargs:
        gen_kwargs["num_return_sequences"] = gen_kwargs.pop("n")

    if "max_tokens" in gen_kwargs:
        gen_kwargs["max_new_tokens"] = gen_kwargs.pop("max_tokens")

    gen_kwargs["do_sample"] = True 
    if (gen_kwargs["top_p"] == None) and (gen_kwargs["temperature"] == 0.0):
        gen_kwargs["top_p"] = None
        gen_kwargs["temperature"] = None
        gen_kwargs["top_k"] = None
        gen_kwargs["do_sample"] = False

    return gen_kwargs


