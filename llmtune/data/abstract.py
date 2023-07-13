from abc import ABC, abstractmethod
from typing import Dict, Any


# Abstract train data loader
class AbstractTrainData(ABC):
    """
    """
    @abstractmethod
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len: int) -> None:
        """
        Args:
            dataset (str): Path to dataset
            val_set_size (int) : Size of validation set
            tokenizer (_type_): Tokenizer
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.val_set_size = val_set_size
        self.cutoff_len = cutoff_len
        self.train_data = None
        self.val_data = None

    @abstractmethod
    def tokenize(self, prompt: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def prepare_data(self) -> None:
        """Loads dataset from file and prepares train_data for trainer."""
        pass
