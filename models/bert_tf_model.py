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

OUTPUT_FOLDER = "saved_models/fine_tuned"
TF_HUB_URL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"  # version 2 of TF Hub pre-trained BERT
# !wget "https://storage.googleapis.com/tfhub-modules/tensorflow/bert_en_uncased_L-12_H-768_A-12/3.tar.gz"

MAX_SEQ_LEN = 512   # Max possible sequence length for BERT base is 512


class BertModel:
    """Bert Model Wrapper.
    This class is a Keras wrapper for a model with a pre-trained BERT layer and a Dense layer used for classification.
    Examples of usage:
        1) Training a new fine-tuned BERT model:
            1.1) model = BertModel() -> Create an instance of the BertModel wrapper
            1.2) model.set_experiment_description("This description string will be saved in the metadata folder") [Optional] -> Set the description information
            1.3) Alternative 1:
                    wget "https://storage.googleapis.com/tfhub-modules/tensorflow/bert_en_uncased_L-12_H-768_A-12/3.tar.gz" -> Download the pre-trained BERT TF Hub model on disk
                    model.load_pre_trained_bert_tf_hub_from_dir("/bert_en_uncased_L-12_H-768_A-12_3") -> Load the pre-trained BERT TF Hub model from disk
                Alternative 2:
                    model.load_pre_trained_bert_tf_hub_from_url("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3") -> Load the pre-trained BERT TF Hub model from URL
            1.4) model.create_model(n_out_units=1, max_seq_length=256, activation_function="softmax", dropout_perc=0.2) -> Create the Keras model
            1.5) model.compile(learning_rate=2e-5) -> compile the model
            1.6) model.fit(train_data, test_data, epochs=4, batch_size=32, verbose=1, n_train=number_train_examples, n_test=number_test_examples) -> train the model
            1.7) model.save()
        2) Load a fine-tuned model from disk for prediction of features extraction
            2.1) model = bert_model.BertModel() -> Create an instance of the BertModel wrapper
            2.2) model.load_model(directory_path) -> Load fine-tuned model from local disk
            2.3) tokenizer = model.get_tokenizer() [Optional] -> Get the tokenizer
                2.4A) Features Extraction (from last BERT layer):
                    2.4A.1) Extract features and save on hdf5 file:
                            features, output_path = model.extract_and_save_embeddings(df["text"].tolist(), df["label"].tolist(), filename="embeddings.hdf5", output_dir="/embeddings_folder")
                    2.4A.2) Extract features:
                            features = model.extract_embeddings(df["text"].tolist(), df["label"].tolist())
                2.4B) Prediction:
                    2.4B.1) predictions = model.predict(["Sentence 1","Sentence 2"])
    """
    def __init__(self):
        """Constructs BertModel.  """
        self.max_seq_length = None
        self.label_list = None
        self.pre_trained_bert_layer = None
        self.tokenizer = None
        self.vocab_file = None
        self.do_lower_case = True
        self.model = None
        self.history = None
        self.flag_compile = False
        self.extractor = None
        self.experiment_description = ""
        self.info_dict = {}
        return

    def load_pre_trained_bert_tf_hub_from_url(self, url=TF_HUB_URL, trainable_flag=True):
        """Loads the TF Hub pre-trained BERT model from URL.
        Args:
            url (str[optional]): URL string of the TF Hub link (default url is stored in TF_HUB_URL)
            trainable_flag (bool[optional]): True if want to train also BERT layer wights (highly suggested), False otherwise. Default is True
        """
        self.pre_trained_bert_layer = hub.KerasLayer(url, trainable=trainable_flag)
        self.pre_trained_bert_layer._name = "bert_layer"
        self.info_dict.update([("trained_bert_weights", trainable_flag)])

        self.vocab_file = self.pre_trained_bert_layer.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = self.pre_trained_bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(self.vocab_file, self.do_lower_case)
        return

    def load_pre_trained_bert_tf_hub_from_dir(self, bert_tf_hub_dir, trainable_flag=True):
        """Loads the TF Hub pre-trained BERT model from local disk.
        The TF Hub module can be downloaded on local disk with the following command:
            !wget "https://storage.googleapis.com/tfhub-modules/tensorflow/bert_en_uncased_L-12_H-768_A-12/bert_en_uncased_L-12_H-768_A-12_2.tar.gz"
            !tar -xvf  '/bert_en_uncased_L-12_H-768_A-12_2.tar.gz' -C 'saved_models/pre_trained/bert_en_uncased_L-12_H-768_A-12_2'
        Args:
            bert_tf_hub_dir (str): PATH string of the TF Hub directory
            trainable_flag (bool[optional]): True if want to train also BERT layer wights (highly suggested), False otherwise. Default is True
        """
        self.pre_trained_bert_layer = hub.KerasLayer(bert_tf_hub_dir, trainable=trainable_flag)
        self.pre_trained_bert_layer._name = "bert_layer"
        self.info_dict.update([("trained_bert_weights", trainable_flag)])

        self.vocab_file = self.pre_trained_bert_layer.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = self.pre_trained_bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = tokenization.FullTokenizer(self.vocab_file, self.do_lower_case)
        return

    def load_tokenizer_from_pickle(self, model_directory, tokenizer_filename="bert_tokenizer.pickle"):
        """ Loads the tokenizer from a saved pickle file.
        This method is an alternative way to load the tokenizer with respect to tokenization.FullTokenizer(vocab_file, do_lower_case=True).
        During training the instance of the FullTokenizer class has been saved into a file named: `bert_tokenizer.pickle`.
        Args:
            model_directory (str): Path of the model's folder
            tokenizer_filename (str): Filename of the saved tokenizer
        Returns:
            tokenization.FullTokenizer() : Instance of the FullTokenizer class
        """
        with open(os.path.join(model_directory, tokenizer_filename), "rb") as tokenizer_file:
            self.tokenizer = pickle.load(tokenizer_file)
        return self.tokenizer

    def load_model(self, model_directory):
        """Loads the fine-tuned model from directory.
        Args:
            model_directory (str): PATH string of the fine-tuned model's directory on local disk
        """
        self.tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(model_directory, "assets", "vocab.txt"), do_lower_case=True)

        print(type(self.tokenizer))

        self.model = keras.models.load_model(model_directory)

        self.max_seq_length = 256

        self.extractor = tf.keras.Model(inputs=self.model.inputs,
                                        outputs=[self.model.get_layer("bert_layer").output])

        self.model.summary()

        return

    def create_model(self, n_out_units, label_list, max_seq_length=None, out_activation=None, dropout_percentage=None):
        """Creates the Keras model.
        Args:
            n_out_units (int): Number of output units in the last layer of the model (to be changed based on the task and number of labels)
            label_list (list[int]): List of integers containing the class label values
            max_seq_length (int[optional]): Maximum number of tokens in the sequence (default is the maximum possible number = 512)
            out_activation (string): String containing the output activation function (e.g. softmax for multiclass, sigmoid for binary) (default is softmax if n_out_units>2, sigmoid if n_out_units=1)
            dropout_percentage (int): Percentage of dropout after the BERT layer (default is 0.2)
        """
        n_out_units, self.label_list, self.max_seq_length, out_activation, dropout_percentage = self.__check_create_model_parameters(n_out_units, label_list, max_seq_length, out_activation, dropout_percentage)

        input_word_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                               name="input_word_ids")

        input_mask = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                           name="input_mask")

        input_type_ids = tf.keras.layers.Input(shape=(self.max_seq_length,), dtype=tf.int32,
                                               name="input_type_ids")

        encoder_inputs = dict(
            input_word_ids=input_word_ids,
            input_mask=input_mask,
            input_type_ids=input_type_ids,
        )

        outputs = self.pre_trained_bert_layer(encoder_inputs)

        pooled_output, sequence_output, encoder_outputs = outputs["pooled_output"], outputs["sequence_output"], outputs["encoder_outputs"]

        drop = tf.keras.layers.Dropout(dropout_percentage)(pooled_output)

        output = tf.keras.layers.Dense(n_out_units, activation=out_activation, name="output")(drop)

        model = tf.keras.Model(
            inputs=encoder_inputs,
            outputs=output
        )

        self.model = model
        self.info_dict["create_model_info"] = [{"number_of_output_units": n_out_units, "label_list": label_list,
                                               "max_sequence_length": max_seq_length, "output_activation_function": out_activation, "dropout_percentage": dropout_percentage}]
        print("INFO: model created ", self.info_dict["create_model_info"])
        return

    def compile(self, learning_rate=2e-5, loss=None, metrics=['accuracy']):
        """Compiles the model.
        Args:
            learning_rate (float[optional]): Learning rate value (default is 2e-5)
            loss:
            metrics:
        """
        learning_rate, loss, metrics = self.__check_compile_parameters(learning_rate, loss, metrics)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=loss,
                           metrics=metrics
                           )

        self.info_dict["compile_info"] = [{"learning_rate": learning_rate, "optimizer": "Adam", "loss_function": str(loss), "metrics": metrics}]
        print("INFO: model compiled ", self.info_dict["compile_info"])
        self.flag_compile = True
        self.model.summary()
        return

    def fit(self, train_data, test_data, epochs, batch_size=32, verbose=1, n_train=None, n_test=None):
        """Fits the model for training.
        Train the model with the examples passed as parameters. Train and Test examples must be of type tensorflow Dataset.
        If have a pandas dataframe, then use: tf.data.Dataset.from_tensor_slices() for the conversion.
        Args:
              train_data (tf.data.Dataset): Training examples as TensorFlow Dataset
              test_data (tf.data.Dataset): Validation examples as TensorFlow Dataset
              epochs (int): Number of epochs for training
              batch_size (int[optional]): Batch size (default is 64)
              verbose (int[optional]): Verbose value in training (default is 1)
              n_train (int[optional]): Number of training examples. This information is stored in the metadata folder after training.
              n_test (int[optional]): Number of test examples. This information is stored in the metadata folder after training.
        Returns:
            tf.keras.callbacks.History(): Training history
        """
        self.__check_fit_parameters()

        with tf.device('/cpu:1'):
            # train
            train_data = (train_data.map(self.__to_feature_map,
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)  # let TensorFlow to automatically set the parallelism on map training texts to features
                          .shuffle(1000)  # Shuffle randomly in batch of 1000
                          .batch(batch_size, drop_remainder=True)  # Training in batch of size specified in `batch_size`. If drop_remainder is True, then the last batch = len_train % batch_size is dropped
                          .prefetch(tf.data.experimental.AUTOTUNE))  # let TensorFlow to automatically set the parallelism on pre-fetching dataset
            # valid
            test_data = (test_data.map(self.__to_feature_map,
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE)  # let TensorFlow to automatically set the parallelism on map testing texts to features
                         .batch(batch_size, drop_remainder=True)  # Testing in batch of size specified in `batch_size`. If drop_remainder is True, then the last batch = len_test % batch_size is dropped
                         .prefetch(tf.data.experimental.AUTOTUNE))  # let TensorFlow to automatically set the parallelism on pre-fetching dataset

        # Train the model
        self.history = self.model.fit(train_data,
                                      validation_data=test_data,
                                      epochs=epochs,
                                      verbose=verbose)

        print("INFO: Training completed.")
        # Store information of training in
        self.info_dict["training_info"] = [{"train_data_shape": str(train_data.element_spec), "test_data_shape": str(test_data.element_spec),
                                           "epochs": epochs, "batch_size": batch_size, "n_train_examples": n_train, "n_test_examples": n_test}]
        return self.history

    def fit_and_save(self, train_data, test_data, epochs, batch_size=64, verbose=1, n_train=None, n_test=None, folder_name=None, experiment_description=None):
        """Fits the model for training and saves the model.
        Train the model with the examples passed as parameters and save it in the specified folder. Train and Test examples must be of type tensorflow Dataset.
        If have a pandas dataframe, then use: tf.data.Dataset.from_tensor_slices() for the conversion.
        Args:
              train_data (tf.data.Dataset): Training examples as TensorFlow Dataset
              test_data (tf.data.Dataset): Validation examples as TensorFlow Dataset
              epochs (int): Number of epochs for training
              batch_size (int[optional]): Batch size (default is 64)
              verbose (int[optional]): Verbose value in training (default is 1)
              n_train (int[optional]): Number of training examples. This information is stored in the metadata folder after training.
              n_test (int[optional]): Number of test examples. This information is stored in the metadata folder after training.
              folder_name (str[optional]): Folder path where will be saved the model
              experiment_description (str[optional]): Description string attached at the end of the folder's filename
        Returns:
            tf.keras.callbacks.History(): Training history
            str: output path where it was saved the model
        """
        history = self.fit(train_data, test_data, batch_size, epochs, verbose, n_train, n_test)

        output_path = self.save_model(folder_name, experiment_description)
        return history, output_path

    def save_model(self, folder_name=None, experiment_description=None):
        """Saves the fine-tuned model and metadata info on local disk.
        The folder name of the saved model is composed as: TIMESTAMP_bert_model[_experiment_description]
        Inside it created a folder named `METADATA` containing:
            TIMESTAMP_history.json: file with the history of training
            experiment_info.json: file containing info about experiment and hyper-parameters used during traning
        Inside it is created a file containing the tokenizer, named `bert_tokenizer.pickle`
        Args:
            folder_name (str[optional]): Folder path where will be saved the model
            experiment_description (str[optional]): Description string attached at the end of the folder's filename
        Returns:
            str : folder path where has been saved the model
        """
        if folder_name is None:
            ts = time.time()
            folder_name = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

        if experiment_description is None:
            out_name = '{}_bert_model'.format(folder_name)
        else:
            out_name = '{}_bert_model_{}'.format(folder_name, experiment_description)

        output_path = os.path.join(OUTPUT_FOLDER, out_name)
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

        with open(os.path.join(output_path, 'bert_tokenizer.pickle'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(output_path, 'metadata', 'experiment_info.json'), 'w') as fp:
            json.dump(self.info_dict, fp)

        print("INFO: Model saved in path: ", output_path)
        return output_path

    def predict(self, input_texts):
        """Predicts output probabilities from input texts.
        Predict output probabilities given a list of raw input texts.
        Args:
            input_texts (list[string]): List of sentences to be predicted
        Returns:
            list(list(float)): list of output probabilities for each input
        """
        tensor_data = tf.data.Dataset.from_tensor_slices((input_texts, [0]*len(input_texts)))

        tensor_batch = (tensor_data.map(self.__to_feature_map).batch(1))

        predictions = self.model.predict(tensor_batch)
        return predictions

    def extract_embeddings(self, input_texts, input_labels, batch_size):
        """Extracts embedding from last BERT layer.
        Extracts an embedding tensor from the last layer of BERT with shape (len(input_texts), 768).
        Args:
            input_texts (list(str)): List of input texts
            input_labels (list(int)): list of labels
            batch_size (int): Size of the batch for the feature extraction
        Returns:
            np.array(len(input_texts, 768)): embedding tensor
        """
        embedding_tensor = np.empty([len(input_texts), 768], dtype=np.float32)
        for i in range(len(input_texts)//batch_size):
            input_slice = input_texts[(i*batch_size):(i+1)*batch_size]

            embedding_tensor[(i*batch_size):(i+1)*batch_size] = self.__extract_single_embedding_batch(input_slice)

        if len(input_texts) % batch_size != 0:
            input_slice = input_texts[:-len(input_texts) % batch_size]

            embedding_tensor[:-len(input_texts) % batch_size] = self.__extract_single_embedding_batch(input_slice)

        return embedding_tensor

    def extract_and_save_embeddings(self, input_texts, input_labels, batch_size, filename, output_folder):
        """Extracts embedding from last BERT layer and save it to hdf5 file.
        Extracts an embedding tensor from the last layer of BERT with shape (len(input_texts), 768) and save it in the specified folder.
        Args:
            input_texts (list(str)): List of input texts
            input_labels (list(int)): list of labels
            batch_size (int): Size of the batch for the feature extraction
            filename (str):
            output_folder (str):
        Returns:
            np.array(len(input_texts, 768)): embedding tensor
        """
        output_path = os.path.join(output_folder, filename)

        fp = h5py.File(output_path, "w")

        embedding_tensor = np.empty([len(input_texts), 768], dtype=np.float32)

        for i in range(len(input_texts) // batch_size):
            input_slice = input_texts[(i * batch_size):(i + 1) * batch_size]

            embedding_tensor[(i * batch_size):(i + 1) * batch_size] = self.__extract_single_embedding_batch(input_slice)

        if len(input_texts) % batch_size != 0:
            input_slice = input_texts[:-len(input_texts) % batch_size]

            embedding_tensor[:-len(input_texts) % batch_size] = self.__extract_single_embedding_batch(input_slice)

        fp.create_dataset("data", data=embedding_tensor, compression="gzip")
        fp.create_dataset("class_id", data=input_labels, compression="gzip")
        fp.close()

        return embedding_tensor, output_path

    def __to_feature(self, text, label):
        """
        Args:
             text (str):
             label (int):
        """
        example = classifier_data_lib.InputExample(guid=None,
                                                   text_a=text.numpy(),
                                                   text_b=None,
                                                   label=label.numpy())

        feature = classifier_data_lib.convert_single_example(0, example, self.label_list, self.max_seq_length, self.tokenizer)

        return feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id

    def __to_feature_map(self, text, label):
        """Express computation in TensorFlow Graph as python function (wraps __to_feature() func).
        """
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

    def __extract_single_embedding_batch(self, input_slice):
        tensor_data = tf.data.Dataset.from_tensor_slices((input_slice, [0] * len(input_slice)))

        tensor_batch = (tensor_data.map(self.__to_feature_map).batch(1))

        batch_embedding_tensor = self.extractor.predict(tensor_batch)

        return batch_embedding_tensor[0][0][:]

    def summary(self):
        self.model.summary()
        return

    def get_model(self):
        """Gets the fine-tuned model.
        Returns:
        """
        return self.model

    def get_tokenizer(self):
        """Gets the tokenizer.
         """
        return self.tokenizer

    def set_experiment_description(self, description):
        """Sets the experiment description string.
        Args:
            description (string): Set the experiment description. This information will be saved in the output metadata folder after training.
        """
        self.experiment_description = description
        self.info_dict["experiment_description"] = description
        return

    def __check_create_model_parameters(self, n_out_units, label_list, max_seq_length, out_activation, dropout_percentage):
        """Checks the parameters of create_model method.
        The model cannot be compiled if it has not been loaded a TF Hub BERT module first (from disk or from URL).
        Raises:
        ValueError: If `self.pre_trained_bert_layer` is `None`. If `self.pre_trained_bert_layer` is None, then the TF Hub BERT module has not yet been loaded.
        """
        if self.pre_trained_bert_layer is None:
            raise ValueError("ERROR: Before create the model is needed to load the tf hub. Try methods load_pre_trained_bert_tf_hub_from_url(url,trainable_flag) or "
                             "load_pre_trained_bert_tf_hub_from_dir(dir,trainable_flag) before invoking the method create_model().")

        # Check consistency between the list of labels provided and the number of output units
        if len(label_list) == 2:
            # If have two labels (binary task) it is suggested to set one single output with a sigmoid
            if n_out_units == 2:
                print("WARNING: is suggested to set one output neuron when have a binary problem.")
            # Inconsistency between label_list and n_out_units
            if n_out_units > 2:
                raise ValueError("ERROR: The number of output neurons not matches the number of labels.")

        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
        else:
            self.max_seq_length = MAX_SEQ_LEN

        self.label_list = label_list

        if len(label_list) > 2 and out_activation is None:
            out_activation = "softmax"
            print("WARNING: Output activation function not specified and automatically set to Softmax.")
        else:
            if len(label_list) <= 2 and out_activation is None:
                out_activation = "sigmoid"
                print("WARNING: Output activation function not specified and automatically set to Sigmoid.")

        if dropout_percentage is None:
            dropout_percentage = 0.2

        return n_out_units, label_list, max_seq_length, out_activation, dropout_percentage

    def __check_compile_parameters(self, learning_rate, loss, metrics):
        """Checks the parameters of compile method.
        The model cannot be compiled if it has not been created first.
        Raises:
        ValueError: If `self.model` is `None`. If `self.model` is None, then the model has not yet been created.
        """
        if self.model is None:
            raise ValueError("ERROR: Before compile the model is needed create the model. Try method create_model(n_out_units, label_list, max_seq_length, out_activation).")

        # If the number of classes is > 2 and the loss function is not specified, then the default loss is `sparse_categorical_crossentropy`
        if loss is None and len(self.label_list) > 2:
            loss = tf.keras.losses.sparse_categorical_crossentropy
            print("WARNING: Loss Function not specified and automatically set to Sparse Categorical Crossentropy.")
        else:
            # If the number of classes is <= 2 and the loss function is not specified, then the default loss is `binary_crossentropy`
            if loss is None and len(self.label_list) <= 2:
                loss = tf.keras.losses.binary_crossentropy
                print("WARNING: Loss Function not specified and automatically set to Binary Crossentropy.")

        return learning_rate, loss, metrics

    def __check_fit_parameters(self):
        """Checks the parameters of fit method.
        The model cannot be trained if it has not been compiled first.
        Raises:
        ValueError: If `self.flag_compile` is `False`. If `self.flag_compile` is `False`, then the model has not yet been compiled.
        """
        if not self.flag_compile:
            raise ValueError("ERROR: Before training the model is needed compile the model. Try method compile(learning_rate, loss, metrics).")
        return