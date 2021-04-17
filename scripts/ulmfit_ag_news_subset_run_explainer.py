import explainer
from model_wrappers import ulmfit_model_wrapper
import re
import os
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from official.nlp.bert import tokenization
from utils import utils
import fastai
from fastai.text import *
from fastai.layers import *





MODEL_DIR = os.path.join("saved_models", "fine_tuned")
USE_CASE_NAME = "ulmfit_model_ag_news_subset_exp0"
FINE_TUNED_MODEL_DIR = os.path.join(utils.get_project_root(), MODEL_DIR, USE_CASE_NAME)

LABEL_LIST = [0, 1, 2, 3]
MAX_SEQ_LENGTH = 256

DATASET_NAME = "ag_news_subset"
DF_TRAIN_PATH = os.path.join(utils.get_project_root(), "datasets", DATASET_NAME, "df_train.csv")
DF_TEST_PATH = os.path.join(utils.get_project_root(), "datasets", DATASET_NAME, "df_test.csv")

def cleaning_function_bert(text):
    return text


def load_model(model_directory):
    """Loads the fine-tuned model from directory.

    Args:
        model_directory (str): PATH string of the fine-tuned model's directory on local disk
    """
    tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(model_directory, "assets", "vocab.txt"),
                                           do_lower_case=True)

    print(type(tokenizer))

    model = keras.models.load_model(model_directory)

    max_seq_length = MAX_SEQ_LENGTH

    extractor = tf.keras.Model(inputs=model.inputs,
                               outputs=[model.get_layer("bert_layer").output])

    model.summary()

    return model, tokenizer, max_seq_length, extractor


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

def process_doc(learn, doc):
    xb, yb = learn.data.one_item(doc)
    return xb

def encode_doc(learn, doc):
    xb = process_doc(learn, doc)

  # Reset initializes the hidden state
    awd_lstm = learn.model[0]
    awd_lstm.reset()
    with torch.no_grad():
        out = awd_lstm.eval()(xb)
    # Return raw output, for last RNN, on last token in sequence
    #return out[0][2][:].max(0).values.detach().numpy()
    return out[0][2].numpy()


if __name__ == "__main__":
    #learn1 = torch.load("/Users/salvatore/PycharmProjects/T-EBAnO-Express/saved_models/fine_tuned/ulmfit_model_cola_exp0/learn1_file.pth")

    #learn_model_dir = "/Users/salvatore/PycharmProjects/T-EBAnO-Express/saved_models/fine_tuned/ulmfit_model_cola_exp0/archive/data.pkl"

    #learn = language_model_learner( pretrained_model=learn_model_dir, drop_mult=0.3)

    #learn1 = load_learner(learn_model_dir)
    learn1 = load_learner("/Users/salvatore/PycharmProjects/T-EBAnO-Express/saved_models/fine_tuned/ulmfit_model_ag_news_subset_exp0/learn1/")

    txt = "carolina 's davis done for the season . charlotte , n.c . ( sports network ) - carolina panthers running back stephen davis will miss the remainder of the season after being placed on injured reserve saturday ."

    help(learn1)

    #learn1 = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)
    #learn1.load_encoder('AIBoot_enc')

    print(type(learn1))

    preds = learn1.predict(txt)

    print(preds)

    out_enc = encode_doc(learn1, txt)

    print(out_enc[0][1])

    print(out_enc.shape)

    tokenizer = SpacyTokenizer(lang="en")

    print(len(tokenizer.tok(txt)))

    model_wrapper = ulmfit_model_wrapper.ULMfitModelWrapper(label_list=LABEL_LIST,
                                                            tokenizer=tokenizer,
                                                            learn1=learn1)

    texts = [txt]*5
    predictions = model_wrapper.predict(texts)

    print(predictions.shape)

    df_test = load_dataset_from_csv(DF_TEST_PATH)

    texts = df_test["text"][100:112].tolist()
    true_labels = df_test["label"][100:112].tolist()

    # Instantiate the LocalExplainer class for the current mdoel
    exp = explainer.LocalExplainer(model_wrapper, model_name=USE_CASE_NAME)

    texts = ["hunting for deer with a mouse . san antonio - forget about playstation 2 - texas entrepreneur wants to kick computer gaming up to the next level by offering players a chance at some real - live killing via mouse and modem ."]

    exp.fit_transform(input_texts=texts,
                      classes_of_interest=[-1] * len(texts),
                      expected_labels=true_labels,
                      flag_pos=True,
                      flag_sen=True,
                      flag_mlwe=True,
                      flag_rnd=False,
                      flag_combinations=True)


    """

    model, tokenizer, max_seq_length, extractor = load_model(FINE_TUNED_MODEL_DIR)

    model_wrapper = ulmfit_model_wrapper.ULMfitModelWrapper(model, extractor, tokenizer,
                                                            label_list=LABEL_LIST,
                                                            max_seq_len=MAX_SEQ_LENGTH,
                                                            clean_function=cleaning_function_bert,
                                                            from_logits=True)

    df_test = load_dataset_from_csv(DF_TEST_PATH)

    texts = df_test["sentence"][:512].tolist()
    true_labels = df_test["label"][:512].tolist()

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


