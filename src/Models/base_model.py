from abc import ABC, abstractmethod
import keras

class BaseModel(ABC):
    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length
        self.n_features = 13
        self.tf_seed = 42

    @abstractmethod
    def build_model(self, hparams: dict) -> keras.Model:
        pass

    @abstractmethod
    def get_space(self, epochs: int, loss_method: str) -> dict:
        pass