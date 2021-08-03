from keras.preprocessing.sequence import pad_sequences
from typing import List
import model_wrapper_interface


class KerasModelWrapper(model_wrapper_interface.ModelWrapperInterface):
    """

    """
    def __init__(self, model, tokenizer, max_seq_len, seq_padding='post', seq_truncating='post', clean_function=None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seq_padding = seq_padding
        self.seq_truncating = seq_truncating
        self.clean_function = clean_function
        self.word_index = self.tokenizer.word_index
        self.index_word = dict(map(reversed, self.word_index.items()))

    def predict(self, input_texts: List[str]) -> list:
        sequences = self.tokenizer.texts_to_sequences(input_texts)
        X_pad = pad_sequences(sequences, padding=self.seq_padding, truncating=self.seq_truncating, maxlen=self.max_seq_len)
        Y = self.model.predict(X_pad)
        return Y

    def clean_text(self, input_text: str) -> str:
        if self.clean_function is None:
            return input_text
        else:
            return self.clean_function(input_text)

    def texts_to_sequences(self, input_texts: List[str]) -> List[List[int]]:
        return self.tokenizer.texts_to_sequences(input_texts)

    def sequences_to_texts(self, sequences: List[List[int]]) -> List[List[str]]:
        return self.tokenizer.sequences_to_texts(sequences)

    def texts_to_tokens(self, input_texts: List[List[str]]) -> List[List[str]]:
        sequences = self.tokenizer.texts_to_sequences(input_texts)
        return [[self.index_word[s] for s in sequence if s != 0] for sequence in sequences]

    def extract_embedding(self):
        pass