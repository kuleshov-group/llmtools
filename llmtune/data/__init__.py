from llmtune.data.text import TrainTxt
from llmtune.data.alpaca import TrainSAD
from llmtune.data.gpt4all import TrainGPT4All

def load_finetuning_data(tune_config, tokenizer):
    if tune_config.ds_type == "alpaca":
        data = TrainSAD(
            tune_config.dataset, 
            tune_config.val_set_size, 
            tokenizer, 
            tune_config.cutoff_len
        )
    elif tune_config.ds_type == "gpt4all":
        raise NotImplementedError('GPT4All dataset currently not supported')
        data = TrainGPT4All(
            tune_config.dataset, 
            tune_config.val_set_size, 
            tokenizer, 
            tune_config.cutoff_len
        )
    else:
        raise ValueError(f"Invalid data name: {tune_config.ds_type}")
    # data.prepare_data(
    #     thd=tune_config.txt_row_thd, use_eos_token=tune_config.use_eos_token
    # )
    data.prepare_data()
    return data