
from trl import SFTConfig, SFTTrainer
from utils.cuda import claim_memory
from src.trl.TRL import TRL

class SFT(TRL):

    def __init__(self, model_agent, training_config, save_dir) -> None:
        super().__init__(model_agent, training_config, save_dir)

    def train(self, dataset_dict, training_args, peft_config=None):
        model = self.model_agent.model
        args = SFTConfig(**training_args)
        
        print("Dataset", dataset_dict)
        print("Arguments", args)
        print("Model", model)
        
        if peft_config is not None:
            print("Peft Config", peft_config)

        trainer = SFTTrainer(
            model=model,
            peft_config=peft_config,
            processing_class=self.model_agent.tokenizer,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
            args=args,
        )
        trainer.train()
        trainer.save_model()
        
        del self.model_agent.model
        del trainer
        claim_memory()