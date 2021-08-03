from keras.preprocessing.sequence import pad_sequences
from typing import List
import model_wrapper_interface
import numpy as np
import torch

class ULMfitModelWrapper(model_wrapper_interface.ModelWrapperInterface):
    """

    """
    def __init__(self, label_list, tokenizer, learn1, embedding_dimension=400, clean_function=(lambda x: x.lower())):
        self.label_list = label_list
        self.embedding_dimension = embedding_dimension
        self.tokenizer = tokenizer
        self.learn1 = learn1
        self.clean_function =clean_function
        self.special_tokens = ['xxunk', 'xxpad', 'xxbos', 'xxeos', 'xxfld', 'xxmaj', 'xxup', 'xxrep', 'xxwrep']
        return

    def get_label_list(self):
        return self.label_list

    def predict(self, input_texts: List[str]) -> list:
        predictions = np.zeros((len(input_texts),len(self.label_list)),dtype=float)
        idx = 0
        for text in input_texts:
            pred = self.learn1.predict(text)
            predictions[idx] = np.array(pred[2].numpy())
            idx += 1
        return predictions

    def clean_text(self, input_text: str) -> str:
        if self.clean_function is None:
            return input_text
        else:
            return self.clean_function(input_text)

    def texts_to_sequences(self, input_texts: List[str]) -> List[List[int]]:
        tokens = []
        for text in input_texts:
            current_tokens = self.tokenizer.tokenizer(text)
            tokens.append(current_tokens)
        return tokens

    def sequences_to_texts(self, sequences: List[List[int]]) -> List[List[str]]:
        return [" ".join(sequence) for sequence in sequences]

    def texts_to_tokens(self, input_texts: List[List[str]]) -> List[List[str]]:
        tokens = []
        for text in input_texts:
            current_tokens = self.tokenizer.tokenizer(text)
            tokens.append(current_tokens)
        return tokens

    def extract_embedding(self, input_texts, batch_size=None):
        list_embedding_tensors = []

        for text in input_texts:
            token_list = self.tokenizer.tokenizer(text)
            current_tokens_embedding_tensor = np.zeros([len(token_list), self.embedding_dimension], dtype=np.float32)

            out_enc = self._encode_text(text)
            current_tokens_embedding_tensor = out_enc[0][1:]
            list_embedding_tensors.append(current_tokens_embedding_tensor)
        return list_embedding_tensors

    def _process_doc(self, doc):
        xb, yb = self.learn1.data.one_item(doc)
        return xb

    def _encode_text(self, doc):
        xb = self._process_doc(doc)

        # Reset initializes the hidden state
        awd_lstm = self.learn1.model[0]
        awd_lstm.reset()
        with torch.no_grad():
            out = awd_lstm.eval()(xb)
        # Return raw output, for last RNN, on last token in sequence
        # return out[0][2][:].max(0).values.detach().numpy()
        return out[0][2].numpy()