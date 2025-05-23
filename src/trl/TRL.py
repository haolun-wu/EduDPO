from peft import LoraConfig

class TRL():

    def __init__(self, model_agent, training_config, save_dir) -> None:
        self.model_agent = model_agent
        self.training_config = training_config
        self.save_dir = save_dir

    def run(self, dataset_dict):
        training_args = self.prepare_training()
        peft_config = self.prepare_peft_config(training_args)

        self.train(dataset_dict, training_args, peft_config)
        
    
    def prepare_training(self):
        """
        Prepare a dictionary of training arguments for the Trainer.

        This is a simplified version of the arguments, which will be
        updated with the training config specific arguments.

        Args:
            None

        Returns:
            base_training_args: A dictionary of training arguments
        """

        batch_size = 1
        gas = 8
        checkpointing = True
            
        base_training_args = {}
        base_training_args["output_dir"] = self.save_dir
        base_training_args["overwrite_output_dir"] = True
        ## Efficient training
        base_training_args["fp16"] = True
        base_training_args["bf16"] = False
        base_training_args["gradient_accumulation_steps"] = gas
        base_training_args["per_device_train_batch_size"] = batch_size
        base_training_args["per_device_eval_batch_size"] = 2 * batch_size
        base_training_args["gradient_checkpointing"] = checkpointing
        base_training_args["use_liger_kernel"] = True 
        base_training_args["max_length"] = 4096
        ## Base training arguments
        base_training_args["num_train_epochs"] = 3
        base_training_args["lr_scheduler_type"] = "cosine"
        base_training_args["max_grad_norm"] = 1.0 # careful, changed 
        base_training_args["warmup_ratio"] = 0.1
        base_training_args["eval_strategy"] = "steps"
        base_training_args["eval_steps"] = 0.1
        base_training_args["save_strategy"] = "steps"
        base_training_args["save_steps"] = 0.1
        base_training_args["logging_strategy"] = "steps"
        base_training_args["logging_steps"] = 5 
        base_training_args["save_total_limit"] = 2
        base_training_args["load_best_model_at_end"] = True 
        ## Bonus 
        # base_training_args["report_to"] = "wandb"

        base_training_args.update(**self.training_config.args)

        return base_training_args

    def prepare_peft_config(self, base_training_args):
        # We are training LORA adapter so we can
        # load the model in any precision we want
        # by default, we loaded the model in fp16
        peft_config = None 
        if self.model_agent.config.lora:
            lora_config = {
                "r": 32, 
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": "all-linear",
                "task_type": "CAUSAL_LM",
            }
            lora_config.update(self.model_agent.config.task.lora)
            peft_config = LoraConfig(**lora_config) 


        # We are training the full model, we have to check 
        # the model intended precision for training        
        if peft_config and peft_config.modules_to_save:
            print("converting base model to float 32 is needed")
            self.model_agent.model = self.model_agent.model.float()
            
        elif self.model_agent.supports_flash_attention:
            base_training_args["bf16"] = True 
            base_training_args["fp16"] = False 

        elif self.model_agent.config.dtype == "fp16":
            base_training_args["fp16"] = True

        if self.model_agent.config.quant == 8:
            base_training_args["fp16"] = False 
            base_training_args["bf16"] = False

        return peft_config 
