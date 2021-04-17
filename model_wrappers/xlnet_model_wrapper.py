from typing import List
import model_wrapper_interface

import numpy as np



class XLNetModelWrapper(model_wrapper_interface.ModelWrapperInterface):
    def __init__(self, xlnet_model, extractor, tokenizer, label_list, max_seq_len, clean_function, embedding_dimension=768, from_logits=False):
        self.model = xlnet_model
        self.extractor = extractor
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.max_seq_length = max_seq_len
        self.clean_function = clean_function
        self.max_wordpieces = self.max_seq_length
        self.embedding_dimension = embedding_dimension
        self.from_logits = from_logits
        self.SPIECE_UNDERLINE = "▁"
        return

    def get_label_list(self):
        return self.label_list

    def predict(self, input_texts: List[str]) -> list:
        return






