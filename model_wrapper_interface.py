from typing import List
import abc


class ModelWrapperInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'predict') and
                callable(subclass.predict) and
                hasattr(subclass, 'clean_text') and
                callable(subclass.clean_text) or
                hasattr(subclass, 'texts_to_sequences') and
                callable(subclass.texts_to_sequences) or
                hasattr(subclass, 'sequences_to_texts') and
                callable(subclass.sequences_to_texts) or
                hasattr(subclass, 'texts_to_tokens') and
                callable(subclass.texts_to_tokens) or
                hasattr(subclass, 'extract_embedding') and
                callable(subclass.extract_embedding) or
                NotImplemented)

    @abc.abstractmethod
    def get_label_list(self) -> List[int]:
        """Returns the list of labels.  """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, input_texts: List[str]) -> list:
        """ Output predictions given a list of input texts. """
        raise NotImplementedError

    @abc.abstractmethod
    def clean_text(self, input_texts: str) -> str:
        """ Clean function for the input texts. """
        pass

    @abc.abstractmethod
    def texts_to_sequences(self, input_texts: List[str]) -> List[List[int]]:
        """ Converts a list of texts into a list of sequences' ids. """
        raise NotImplementedError

    @abc.abstractmethod
    def sequences_to_texts(self, sequences: List[List[int]]) -> List[List[str]]:
        """ Converts a list of sequences' ids into a list of string. """
        raise NotImplementedError

    @abc.abstractmethod
    def texts_to_tokens(self, input_texts: List[List[str]]) -> List[List[str]]:
        """ Converts a list of texts into a list of lists of tokens. """
        raise NotImplementedError

    @abc.abstractmethod
    def extract_embedding(self, input_texts, batch_size, layers, layers_aggregation_function):
        """ Extract embedding for each input tokens or word. It should be adapted to the specific model under analysis. """
        raise NotImplementedError
