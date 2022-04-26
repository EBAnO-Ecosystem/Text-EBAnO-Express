from typing import List
import model_wrapper_interface

from official.nlp.data import classifier_data_lib

import tensorflow as tf
import numpy as np
import re


class AlbertTensorFlowModelWrapper(model_wrapper_interface.ModelWrapperInterface):
    def __init__(self, albert_model, extractor, tokenizer, label_list, max_seq_len, clean_function, embedding_dimension=768, from_logits=False):
        self.model = albert_model
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
        tensor_data = tf.data.Dataset.from_tensor_slices((input_texts, [0]*len(input_texts)))

        tensor_batch = (tensor_data.map(self.__to_feature_map).batch(1))

        predictions = self.model.predict(tensor_batch)

        if len(self.label_list) == 2 and self.from_logits is False:
            predictions = np.append(1-predictions, predictions, axis=1).reshape(len(input_texts),2)
        return predictions

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
        #texts = [" ".join(token_list) for token_list in tokens]
        texts = [''.join(token_list).replace(self.SPIECE_UNDERLINE, ' ')[1:] for token_list in tokens]
        #return [re.sub(" ##", "", text) for text in texts]
        return texts

    def texts_to_tokens(self, input_texts: List[List[str]]) -> List[List[str]]:

        # tokens = self.tokenizer.basic_tokenizer.tokenize(input_text)

        # n_wordpieces = len(wordpieces)  # Number of wordpieces in the current input
        # n_tokens = len(tokens)  # Number of full tokens in the current input

        tokens_list = []
        for input_text in input_texts:
            wordpieces = self.tokenizer.tokenize(input_text)
            n_wordpieces_max = min(len(wordpieces), self.max_wordpieces)
            n_tokens_max = len([wordpiece for wordpiece in wordpieces[:n_wordpieces_max] if wordpiece.startswith(self.SPIECE_UNDERLINE) is True])
            #tokens = self.tokenizer.basic_tokenizer.tokenize(input_text)[:n_tokens_max]
            wordpieces_raw = self.tokenizer.tokenize(input_text)[:n_wordpieces_max]

            tokens = []
            index = 0
            for token in wordpieces_raw:
                if token.startswith(self.SPIECE_UNDERLINE):
                    tokens.append(token.replace(self.SPIECE_UNDERLINE, ' '))
                    index += 1
                else:
                    tokens[index-1] = tokens[index-1]+token

            tokens_list.append(tokens)
        return tokens_list

    def extract_embedding(self, input_texts, batch_size, layers=[8, 9, 10, 11], layers_aggregation_function="avg"):

        wordpieces_embedding_tensor = self.__extract_wordpieces_embedding(input_texts, batch_size, layers)

        if layers_aggregation_function == "sum":
            aggregated_embedding = np.sum(wordpieces_embedding_tensor, axis=0)
        if layers_aggregation_function == "avg":
            aggregated_embedding = np.mean(wordpieces_embedding_tensor, axis=0)

        list_embedding_tensors = []

        for input_text, current_full_embedding in zip(input_texts, aggregated_embedding):
            wordpieces = self.tokenizer.tokenize(input_text)
            # tokens = self.tokenizer.basic_tokenizer.tokenize(input_text)

            # n_wordpieces = len(wordpieces)  # Number of wordpieces in the current input
            # n_tokens = len(tokens)  # Number of full tokens in the current input

            current_embedding = current_full_embedding[1:self.max_wordpieces + 1]

            n_wordpieces_max = min(len(wordpieces), self.max_wordpieces)
            n_tokens_max = len(
                [wordpiece for wordpiece in wordpieces[:n_wordpieces_max] if wordpiece.startswith(self.SPIECE_UNDERLINE) is True])

            current_tokens_embedding_tensor = np.zeros([n_tokens_max, self.embedding_dimension], dtype=np.float32)

            # Construct an embedding tensor for full tokens starting from embedding tensor for wordpieces
            t_index = 0  # Token index
            outer_wp_index = 0
            while outer_wp_index < n_wordpieces_max:
                outer_wp = wordpieces[outer_wp_index]
                sum_embedding_accumulator = current_embedding[outer_wp_index]
                count_embedding_accumulator = 1
                for inner_wp_index in range(outer_wp_index + 1, n_wordpieces_max):
                    inner_wp = wordpieces[inner_wp_index]
                    if inner_wp.startswith(self.SPIECE_UNDERLINE) is False:
                        sum_embedding_accumulator = sum_embedding_accumulator + current_embedding[inner_wp_index]
                        count_embedding_accumulator += 1
                    else:
                        break

                # The embedding vectors of wordpices belongin the same full tokens are averaged
                sum_embedding_accumulator = sum_embedding_accumulator / count_embedding_accumulator
                current_tokens_embedding_tensor[t_index] = sum_embedding_accumulator

                # Increment indices
                t_index += 1
                outer_wp_index = outer_wp_index + count_embedding_accumulator

            list_embedding_tensors.append(current_tokens_embedding_tensor)

        return list_embedding_tensors

    def __extract_wordpieces_embedding(self, input_texts, batch_size, layers):
        wordpieces_embedding_tensor = np.empty([len(layers), len(input_texts), self.max_seq_length, self.embedding_dimension],
                                               dtype=np.float32)

        # Extract embedding for each batch
        for i in range(len(input_texts) // batch_size):
            input_slice = input_texts[(i * batch_size):(i + 1) * batch_size]

            wordpieces_embedding_tensor[:][
            (i * batch_size):(i + 1) * batch_size] = self.__extract_single_wordpieces_embedding_batch(input_slice,
                                                                                                      layers)

        # Extract embedding for the remainder batch
        if len(input_texts) % batch_size != 0:
            input_slice = input_texts[:-len(input_texts) % batch_size]

            wordpieces_embedding_tensor[:][
            :-len(input_texts) % batch_size] = self.__extract_single_wordpieces_embedding_batch(input_slice, layers)

        return wordpieces_embedding_tensor

    def __extract_single_wordpieces_embedding_batch(self, input_slice, layers):
        tensor_data = tf.data.Dataset.from_tensor_slices((input_slice, [0] * len(input_slice)))

        tensor_batch = (tensor_data.map(self.__to_feature_map).batch(1))

        batch_embedding_tensor = self.extractor.predict(tensor_batch)

        return [batch_embedding_tensor[0]["encoder_outputs"][layer] for layer in layers]

        #return [batch_embedding_tensor[0]["sequence_output"]]

    def __to_feature(self, text, label):
        example = classifier_data_lib.InputExample(guid=None,
                                                   text_a=text.numpy(),
                                                   text_b=None,
                                                   label=label.numpy())

        feature = classifier_data_lib.convert_single_example(0, example, self.label_list, self.max_seq_length,
                                                             self.tokenizer)

        return feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id

    def __to_feature_map(self, text, label):
        input_ids, input_mask, segment_ids, label_id = tf.py_function(self.__to_feature,
                                                                      inp=[text, label],
                                                                      Tout=[tf.int32, tf.int32, tf.int32, tf.int32])

        input_ids.set_shape([self.max_seq_length])
        input_mask.set_shape([self.max_seq_length])
        segment_ids.set_shape([self.max_seq_length])
        label_id.set_shape([])

        # Input is a dict with `input_word_ids` , `input_mask` , `input_type_ids`
        x = {
            'input_word_ids': input_ids,
            'input_mask': input_mask,
            'input_type_ids': segment_ids
        }

        return x, label_id

