import os

class Finetune4bConfig:
    """Config holder for 4bit finetuning"""
    def __init__(
        self, dataset: str, ds_type: str, 
        lora_out_dir: str,
        mbatch_size: int, batch_size: int,
        epochs: int, lr: float, 
        cutoff_len: int,
        lora_r: int, lora_alpha: int, lora_dropout: float,
        val_set_size: float,
        warmup_steps: int, save_steps: int, 
        save_total_limit: int, logging_steps: int,
     ):
        self.dataset = dataset
        self.ds_type = ds_type
        self.lora_out_dir = lora_out_dir
        self.mbatch_size = mbatch_size
        self.batch_size = batch_size
        self.gradient_accumulation_steps = self.batch_size // self.mbatch_size
        self.epochs = epochs
        self.lr = lr
        self.cutoff_len = cutoff_len
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        # self.lora_dropout = 0 if gradient_checkpointing else lora_dropout
        self.lora_dropout = lora_dropout
        self.val_set_size = int(val_set_size) if val_set_size > 1.0 else float(val_set_size)
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.logging_steps = logging_steps
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.ddp = self.world_size != 1
        self.device_map = "auto" if not self.ddp else {"": self.local_rank}
        if self.ddp:
            self.gradient_accumulation_steps = self.gradient_accumulation_steps // self.world_size