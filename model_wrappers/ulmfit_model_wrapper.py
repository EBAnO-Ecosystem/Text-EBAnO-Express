from keras.preprocessing.sequence import pad_sequences
from typing import List
import model_wrapper_interface
import numpy as np
import torch

class ULMfitModelWrapper(model_wrapper_interface.ModelWrapperInterface):
    """

    """
    def __init__(self, label_list, learn, learn1, embedding_dimension=400):
        self.label_list = label_list
        self.embedding_dimension = embedding_dimension
        self.learn = learn
        self.learn1 = learn1
        self.special_tokens = ['xxunk', 'xxpad', 'xxbos', 'xxeos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep']
        return

    def get_label_list(self):
        return self.label_list

    def predict(self, input_texts: List[str]) -> list:
        return

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

    def _process_doc(self, doc):
        xb, yb = self.learn.data.one_item(doc)
        return xb

    def _encode_text(self, doc):
        xb = self._process_doc(self.learn, doc)

        # Reset initializes the hidden state
        awd_lstm = self.learn.model[0]
        awd_lstm.reset()
        with torch.no_grad():
            out = awd_lstm.eval()(xb)
        # Return raw output, for last RNN, on last token in sequence
        # return out[0][2][:].max(0).values.detach().numpy()
        return out[0][2].numpy()