import explainer
from model_wrappers import albert_model_wrapper
import re
import os
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from utils import utils
from official.nlp.bert import tokenization
import tensorflow_hub as hub


MODEL_DIR = os.path.join("saved_models", "fine_tuned")
USE_CASE_NAME = "albert_model_cola_exp0"
FINE_TUNED_MODEL_DIR = os.path.join(utils.get_project_root(), MODEL_DIR, USE_CASE_NAME)

LABEL_LIST = [0, 1]
LABEL_NAMES = ["Unacceptable", "Acceptable"]
MAX_SEQ_LENGTH = 128

DATASET_NAME = "cola"
DF_TRAIN_PATH = os.path.join(utils.get_project_root(), "datasets", DATASET_NAME, "df_train.csv")
DF_TEST_PATH = os.path.join(utils.get_project_root(), "datasets", DATASET_NAME, "df_test.csv")


def cleaning_function_bert(text):
    return text


def load_model(model_directory):
    """Loads the fine-tuned model from directory.

    Args:
        model_directory (str): PATH string of the fine-tuned model's directory on local disk
    """
    #tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(model_directory, "assets", "vocab.txt"), do_lower_case=True)

    #with open(os.path.join(model_directory, "albert_tokenizer.pickle"), 'rb') as f:
    #    tokenizer = pickle.load(f)





    model = keras.models.load_model(model_directory)

    extractor = tf.keras.Model(inputs=model.inputs,
                               outputs=[model.get_layer("albert_layer").output])

    pretrained_albert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/albert_en_base/2", trainable=False)
    sp_model_file = pretrained_albert_layer.resolved_object.sp_model_file.asset_path.numpy()
    tokenizer = tokenization.FullSentencePieceTokenizer(sp_model_file)

    print(type(tokenizer))

    model.summary()

    return model, tokenizer, extractor



def load_imdb_dataset(train_slice, test_slice, as_supervised=False):
    (train_data, test_data), info = tfds.load('imdb_reviews',
                                              split=['train[{}]'.format(train_slice), 'test[{}]'.format(test_slice)],
                                              as_supervised=as_supervised,
                                              with_info=True
                                              )

    return train_data, test_data


def load_dataset_from_csv(dataset_path):
    df = pd.read_csv(dataset_path)
    return df


if __name__ == "__main__":
    model, tokenizer, extractor = load_model(FINE_TUNED_MODEL_DIR)

    model_wrapper = albert_model_wrapper.AlbertModelWrapper(model, extractor, tokenizer, label_list=LABEL_LIST,
                                                            max_seq_len=MAX_SEQ_LENGTH,
                                                            clean_function=cleaning_function_bert,
                                                            from_logits=True)

    df_test = load_dataset_from_csv(DF_TEST_PATH)

    texts = df_test["sentence"][:512].tolist()
    true_labels = df_test["label"][:512].tolist()


    df_labels = pd.DataFrame(true_labels)
    print(df_labels.describe)


    """
    texts = [text.lower() for text in texts]

    tokens = model_wrapper.texts_to_tokens(texts)

    print(tokens)

    sequences = model_wrapper.texts_to_sequences(texts)

    print(sequences)

    texts_from_seq = model_wrapper.sequences_to_texts(sequences)

    print(texts_from_seq)

    # embeddings = model_wrapper.extract_embedding(input_texts=texts, batch_size=32)

    # print(embeddings)

    # Instantiate the LocalExplainer class for the current mdoel
    exp = explainer.LocalExplainer(model_wrapper, model_name=USE_CASE_NAME)

    exp.fit_transform(input_texts=texts,
                      classes_of_interest=[-1] * len(texts),
                      expected_labels=true_labels,
                      flag_pos=True,
                      flag_sen=True,
                      flag_mlwe=True,
                      flag_rnd=False,
                      flag_combinations=True)

    # exp.fit(input_texts, [1])"""


