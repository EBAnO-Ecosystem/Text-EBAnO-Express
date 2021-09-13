import explainer
from model_wrappers import bert_model_wrapper
import re
import os
import keras
import tensorflow as tf
import pandas as pd
from utils import utils
from official.nlp.bert import tokenization

import time

MODEL_DIR = os.path.join("saved_models", "fine_tuned")
USE_CASE_NAME = "20201128_bert_model_ag_news_subset_exp0"
FINE_TUNED_MODEL_DIR = os.path.join(utils.get_project_root(), MODEL_DIR, USE_CASE_NAME)

LABEL_LIST = [0, 1, 2, 3]
MAX_SEQ_LENGTH = 256

DATASET_NAME = "ag_news_subset"
DF_TRAIN_PATH = os.path.join(utils.get_project_root(), "datasets", DATASET_NAME, "df_train.csv")
DF_TEST_PATH = os.path.join(utils.get_project_root(), "datasets", DATASET_NAME, "df_test.csv")


def cleaning_function_bert(text):
    text = re.sub("@\S+", " ", text) # Remove Mentions
    text = re.sub("https*\S+", " ", text) # Remove URL
    text = re.sub("#\S+", " ", text) # Remove Hastags
    text = re.sub('&lt;/?[a-z]+&gt;', '', text) # Remove special Charaters
    text = re.sub('#39', ' ', text) # Remove special Charaters
    text = re.sub('<.*?>', '', text) # Remove html
    text = re.sub(' +', ' ', text) # Merge multiple blank spaces
    text = text.replace("<br>", "")
    text = text.replace("</br>", "")
    return text


def load_dataset_from_csv(dataset_path):
    df = pd.read_csv(dataset_path)
    return df


def load_model(model_directory):
    """Loads the fine-tuned model from directory.

    Args:
        model_directory (str): PATH string of the fine-tuned model's directory on local disk
    """
    tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(model_directory, "assets", "vocab.txt"), do_lower_case=True)

    print(type(tokenizer))

    model = keras.models.load_model(model_directory)

    extractor = tf.keras.Model(inputs=model.inputs,
                               outputs=[model.get_layer("bert_layer").output])

    model.summary()

    return model, tokenizer, extractor


if __name__ == "__main__":

    input_texts = ["You may fuck you think that azz!!!! abstract methods can't be implemented in the abstract base class. " 
                 "This impression is wrong: An abstract method can have an implementation in the abstract class!" 
                 " Even if they are implemented, designers of subclasses will be forced to override the implementation." 
                 " Like in other cases of normal inheritance, the abstract method can be invoked with super() call mechanism. " 
                 "This enables providing some basic functionality in the abstract method, which can be enriched by the subclass implementation."]

    # Load fine-tuned model from directory
    model, tokenizer, extractor = load_model(FINE_TUNED_MODEL_DIR)

    model_wrapper = bert_model_wrapper.BertModelWrapper(model, extractor, tokenizer, label_list=LABEL_LIST, max_seq_len=MAX_SEQ_LENGTH, clean_function=cleaning_function_bert)

    df_test = load_dataset_from_csv(DF_TEST_PATH)

    #texts = df_test["text"][100:612].tolist()
    #true_labels = df_test["label"][100:612].tolist()

    texts = df_test["text"][:1].tolist()
    true_labels = df_test["label"][:1].tolist()

    #embeddings = model_wrapper.extract_embedding(input_texts=texts, batch_size=32)

    #print(embeddings)

    # Instantiate the LocalExplainer class for the current mdoel
    exp = explainer.LocalExplainer(model_wrapper, model_name=USE_CASE_NAME)

    start_time = time.time()

    exp.fit_transform(input_texts=texts,
                      classes_of_interest=[-1]*len(texts),
                      expected_labels=true_labels,
                      flag_pos=False,
                      flag_sen=False,
                      flag_mlwe=False,
                      flag_rnd=True,
                      flag_combinations=False)

    print("Local Explainers takes {} seconds".format(time.time()-start_time))

    #exp.fit(input_texts, [1])


