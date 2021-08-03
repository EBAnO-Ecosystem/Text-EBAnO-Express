from keras.preprocessing.sequence import pad_sequences
from typing import List
import model_wrapper_interface
import numpy as np

class KerasGloveModelWrapper(model_wrapper_interface.ModelWrapperInterface):
    """

    """
    def __init__(self, model, glove_embedding, tokenizer, label_list, max_seq_len, seq_padding='post',
                 seq_truncating='post', clean_function=None, embedding_dimension=300, from_logits=False):
        self.model = model
        self.glove_embedding = glove_embedding
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.max_seq_len = max_seq_len
        self.seq_padding = seq_padding
        self.seq_truncating = seq_truncating
        self.clean_function = clean_function
        self.word_index = self.tokenizer.word_index
        self.index_word = dict(map(reversed, self.word_index.items()))
        self.embedding_dimension = embedding_dimension
        self.from_logits = from_logits
        return

    def get_label_list(self):
        return self.label_list

    def predict(self, input_texts: List[str]) -> list:
        sequences = self.tokenizer.texts_to_sequences(input_texts)
        X_pad = pad_sequences(sequences, padding=self.seq_padding, truncating=self.seq_truncating, maxlen=self.max_seq_len)
        Y = self.model.predict(X_pad)

        if len(self.label_list) == 2 and self.from_logits is False:
            predictions = np.append(1-Y, Y, axis=1).reshape(len(input_texts),2)

        return predictions

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

    def extract_embedding(self, input_texts, batch_size=None):
        list_embedding_tensors = []

        for token_list in self.texts_to_tokens(input_texts):
            current_tokens_embedding_tensor = np.zeros([len(token_list), self.embedding_dimension], dtype=np.float32)
            idx = 0
            for token in token_list:
                if token == '<OOV>':
                    embedding_vector = np.full((1, self.embedding_dimension), 0)
                else:
                    if token in self.glove_embedding:
                        embedding_vector = self.glove_embedding[token]
                    else:
                        embedding_vector = np.full((1, self.embedding_dimension), 0)
                current_tokens_embedding_tensor[idx] = embedding_vector
                idx += 1

            list_embedding_tensors.append(current_tokens_embedding_tensor)
        return list_embedding_tensors