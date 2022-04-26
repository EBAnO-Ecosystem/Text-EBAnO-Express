import tensorflow as tf
import tensorflow_hub as hub

from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization

import pandas as pd
import numpy as np

import keras
import pickle
import os
import json
import time
import datetime
import h5py


class AlbertModel:
    def __init__(self):

        self.pretrained_albert_layer = None
        self.sp_model_file = None
        self.tokenizer = None
        self.num_classes = None
        self.max_seq_length = None
        self.label_list = None
        self.TF_HUB_URL = "https://tfhub.dev/tensorflow/albert_en_base/2"

        self.info_dict = {}
        return

    def load_pretrained_albert_tf_hub_from_url(self, url=None, trainable_flag=True):
        if url is None:
            url = self.TF_HUB_URL
        self.pretrained_albert_layer = hub.KerasLayer(url, trainable=trainable_flag)
        self.pretrained_albert_layer._name = "albert_layer"

        self.info_dict.update({"trained_albert_weights": trainable_flag})
        return

    def load_pretrained_albert_tf_hub_from_dir(self, tf_hub_model_dir, trainable_flag=True):
        self.pretrained_albert_layer = hub.KerasLayer(tf_hub_model_dir, trainable=trainable_flag)
        self.pretrained_albert_layer._name = "albert_layer"

        self.info_dict.update({"trained_albert_weights": trainable_flag})
        return

    def load_model(self, model_directory):
        # Load Tokenizer
        with open(os.path.join(model_directory, "albert_tokenizer.pickle"), "rb") as tokenizer_file:
            self.tokenizer = pickle.load(tokenizer_file)

        # Load Model
        self.model = keras.models.load_model(model_directory)

        # Read from metadata file the max_seq_length
        with open(os.path.join(model_directory, "metadata", "experimenti_info.json"), "rb") as metadata_file:
            self.info_dict = json.load(metadata_file)
            self.max_seq_length = self.info_dict["max_seq_length"]

        # Load the feature extractor (albert layer)
        self.extractor = tf.keras.Model(inputs=self.model.inputs,
                                        outputs=[self.model.get_layer("albert_layer").output])

        print("INFO: model loaded")
        self.model.summary()
        print("INFO: features extractor loaded")
        self.extractor.summary()
        return

    def create_model(self, num_classes, label_list, label_names=None, dropout_percentage=0.2, max_seq_length=256):

        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.label_list = label_list

        if self.num_classes > 2:
            activation_function = "softmax"
        else:
            activation_function = "sigmoid"

        # Save parameters setting
        self.info_dict.update({"num_classes": num_classes, "label_list": label_list, "label_names": label_names,
                               "dropout_percentage": dropout_percentage, "max_seq_length": max_seq_length,
                               "output_activation": activation_function})

        # Define Albert inputs
        input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                          name="input_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")

        albert_inputs = dict(input_word_ids=input_ids,
                             input_mask=input_mask,
                             input_type_ids=segment_ids)

        # Define Albert block
        albert_outputs = self.pretrained_albert_layer(albert_inputs)

        self.sp_model_file = self.pretrained_albert_layer.resolved_object.sp_model_file.asset_path.numpy()
        self.tokenizer = tokenization.FullSentencePieceTokenizer(self.sp_model_file)

        # pooled_output = albert_outputs["pooled_output"]
        # sequence_output = albert_outputs["sequence_output"]

        drop = tf.keras.layers.Dropout(dropout_percentage)(albert_outputs["pooled_output"])

        classification_output = tf.keras.layers.Dense(num_classes, activation=activation_function,
                                                      name="classification")(drop)

        model = tf.keras.Model(
            inputs=albert_inputs,
            outputs=classification_output
        )

        self.model = model

        return

    def fit(self, train_data, test_data, epochs, batch_size, shuffle_batch_size=10000, verbose=1, drop_remainder=False):

        self.info_dict.update({"epochs": epochs, "batch_size": batch_size, "shuffle_batch_size": shuffle_batch_size})

        with tf.device('/cpu:1'):
            # train
            train_data = (train_data.map(self.to_feature_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                          .shuffle(shuffle_batch_size)
                          .batch(batch_size, drop_remainder=drop_remainder)
                          .prefetch(tf.data.experimental.AUTOTUNE))
            # valid
            test_data = (test_data.map(self.to_feature_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                         .batch(batch_size, drop_remainder=drop_remainder)
                         .prefetch(tf.data.experimental.AUTOTUNE))

        self.history = self.model.fit(train_data,
                                      validation_data=test_data,
                                      epochs=epochs,
                                      verbose=verbose)

        return self.history

    def compile(self, learning_rate=2e-5, loss="sparse_categorical_crossentropy", metrics=['accuracy']):
        self.info_dict.update({"learning_rate": learning_rate, "loss": loss.__str__(), "metrics": metrics})

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           # self.model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=learning_rate),
                           loss=loss,
                           metrics=metrics
                           )

        return

    def to_feature(self, text, label):
        example = classifier_data_lib.InputExample(guid=None,
                                                   text_a=text.numpy(),
                                                   text_b=None,
                                                   label=label.numpy())

        feature = classifier_data_lib.convert_single_example(0, example, self.label_list, self.max_seq_length,
                                                             self.tokenizer)

        return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)

    def to_feature_map(self, text, label):
        input_ids, input_mask, segment_ids, label_id = tf.py_function(self.to_feature,
                                                                      inp=[text, label],
                                                                      Tout=[tf.int32, tf.int32, tf.int32, tf.int32])

        input_ids.set_shape([self.max_seq_length])
        input_mask.set_shape([self.max_seq_length])
        segment_ids.set_shape([self.max_seq_length])
        label_id.set_shape([])

        x = {
            'input_word_ids': input_ids,
            'input_mask': input_mask,
            'input_type_ids': segment_ids
        }

        return (x, label_id)

    def save_model(self, folder_name, experiment_description=None):
        if folder_name is None:
            ts = time.time()
            folder_name = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

        if experiment_description is None:
            out_name = '{}_albert_model'.format(folder_name)
        else:
            out_name = '{}_albert_model_{}'.format(folder_name, experiment_description)

        output_path = os.path.join(folder_name, out_name)
        metadata_path = os.path.join(output_path, "metadata")
        try:
            os.mkdir(output_path)
            os.mkdir(metadata_path)
        except OSError:
            print("Creation of the output directory %s failed" % output_path)

        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(self.history.history)

        # save to json:
        hist_json_file = '{}_history.json'.format(folder_name)
        hist_json_fie_path = os.path.join(metadata_path, hist_json_file)
        with open(hist_json_fie_path, mode='w+') as f:
            hist_df.to_json(f)

        self.model.save(output_path)

        with open(os.path.join(output_path, 'albert_tokenizer.pickle'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(output_path, 'metadata', 'experiment_info.json'), 'w') as fp:
            json.dump(self.info_dict, fp)

        print("INFO: Model saved in path: ", output_path)
        return output_path

    def summary(self):
        self.model.summary()
        return

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_extractor(self):
        return self.extractor


