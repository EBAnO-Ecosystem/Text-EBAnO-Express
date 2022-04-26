import re
from typing import List

import model_wrapper_interface
import numpy as np


class BertModelWrapper(model_wrapper_interface.ModelWrapperInterface):
    def __init__(self, model, tokenizer, clean_function=None, max_seq_length=256, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = list(range(model.num_labels))
        self.max_seq_length = max_seq_length
        if clean_function is None:
            self.clean_function = self.clean_function
        else:
            self.clean_function = clean_function
        self.max_wordpieces = self.max_seq_length
        self.batch_size = batch_size
        return

    def empty_clean_fn(self, text):
        return text

    def predictSingleBatch(self,text):
        inputs = self.tokenizer(text,padding=True, truncation=True, return_tensors="pt",max_length=256)
        outputs = self.model(**inputs)
        probs = outputs[0].softmax(1)
        return np.array(probs.detach())

    def get_label_list(self):
        return self.label_list

    def predict(self, input_texts: List[str]) -> list:
        if isinstance(input_texts, str):
            return self.predictSingleBatch(self.clean_function(input_texts))
        input_texts = list(map(self.clean_function,input_texts))
        result = np.empty((len(input_texts), len(self.label_list)), dtype="float32")
        i = 0
        while 1:
            if i >= len(input_texts):
                break
            j = i + self.batch_size;
            if j > len(input_texts):
                j = len(input_texts)

            result[i:j] = self.predictSingleBatch(input_texts[i:j])
            i = j
        return result

    def clean_text(self, input_text: str) -> str:
        if self.clean_function is None:
            return input_text
        else:
            return self.clean_function(input_text)

    def texts_to_sequences(self, input_texts: List[str]) -> List[List[int]]:
        tokens = [self.tokenizer.tokenize(input_text) for input_text in input_texts]
        word_ids = [self.tokenizer.convert_tokens_to_ids(token_list) for token_list in tokens]
        return [word_id[:min(len(word_id), self.max_wordpieces)] for word_id in word_ids]

    def sequences_to_texts(self, sequences: List[List[int]]) -> List[List[str]]:
        tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in sequences]
        texts = [" ".join(token_list) for token_list in tokens]
        return [re.sub(" ##", "", text) for text in texts]

    def texts_to_tokens(self, input_texts: List[List[str]]) -> List[List[str]]:
        inputs = self.tokenizer(input_texts, padding=False, truncation=True)
        tokens_list = []
        for input_text in input_texts:
            wordpieces = self.tokenizer.tokenize(input_text)
            n_wordpieces_max = min(len(wordpieces), self.max_wordpieces)
            n_tokens_max = len(
                [wordpiece for wordpiece in wordpieces[:n_wordpieces_max] if wordpiece.startswith("##") is False])
            tokens = []
            j=0
            for i in range(n_tokens_max):
                wordpiece = wordpieces[j]
                while j<n_wordpieces_max-1:
                    j = j + 1
                    if wordpieces[j].startswith("##"):
                        wordpiece = wordpiece + wordpieces[j].replace("##","")
                    else:
                        break
                tokens.append(wordpiece)
            tokens_list.append(tokens)
        return tokens_list

    def extract_embedding(self, input_texts, batch_size, layers=[8, 9, 10, 11], layers_aggregation_function="avg"):
        inputs = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states

        a = []
        for i in layers:
            a.append(hidden_states[i].detach().__array__())
        wordpieces_embedding_tensor = np.array(a)
        if layers_aggregation_function == "sum":
            aggregated_embedding = np.sum(wordpieces_embedding_tensor, axis=0)
        if layers_aggregation_function == "avg":
            aggregated_embedding = np.mean(wordpieces_embedding_tensor, axis=0)

        list_embedding_tensors = []

        for input_text, current_full_embedding in zip(input_texts, aggregated_embedding):
            wordpieces = self.tokenizer.tokenize(input_text)

            current_embedding = current_full_embedding[1:self.max_wordpieces+1]

            n_wordpieces_max = min(len(wordpieces), self.max_wordpieces)
            n_tokens_max = len([wordpiece for wordpiece in wordpieces[:n_wordpieces_max] if wordpiece.startswith("##") is False])

            current_tokens_embedding_tensor = np.zeros([n_tokens_max, 768], dtype=np.float32)

            # Construct an embedding tensor for full tokens starting from embedding tensor for wordpieces
            t_index = 0  # Token index
            outer_wp_index = 0
            while outer_wp_index < n_wordpieces_max-1: #aggiunto io
                outer_wp = wordpieces[outer_wp_index]
                sum_embedding_accumulator = current_embedding[outer_wp_index]
                count_embedding_accumulator = 1
                for inner_wp_index in range(outer_wp_index+1, n_wordpieces_max):
                    inner_wp = wordpieces[inner_wp_index]
                    if inner_wp.startswith("##"):
                        sum_embedding_accumulator = sum_embedding_accumulator + current_embedding[inner_wp_index]
                        count_embedding_accumulator += 1
                    else:
                        break

                # The embedding vectors of wordpices belongin the same full tokens are averaged
                sum_embedding_accumulator = sum_embedding_accumulator/count_embedding_accumulator
                current_tokens_embedding_tensor[t_index] = sum_embedding_accumulator

                # Increment indices
                t_index += 1
                outer_wp_index = outer_wp_index + count_embedding_accumulator

            list_embedding_tensors.append(current_tokens_embedding_tensor)

        return list_embedding_tensors