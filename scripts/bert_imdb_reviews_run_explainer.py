from model_wrappers import bert_tensorflow_model_wrapper
import re
import os
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from official.nlp.bert import tokenization
import explainer
import time


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


def load_model(model_directory):
    """Loads the fine-tuned model from directory.

    Args:
        model_directory (str): PATH string of the fine-tuned model's directory on local disk
    """
    tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(model_directory, "assets", "vocab.txt"), do_lower_case=True)

    print(type(tokenizer))

    model = keras.models.load_model(model_directory)

    max_seq_length = 256

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


if __name__ == "__main__":

    model_dir = "../saved_models/fine_tuned/20201117_bert_model_imdb_reviews_exp_0"

    model, tokenizer, max_seq_length, extractor = load_model(model_dir)

    model_wrapper = bert_tensorflow_model_wrapper.BertTensorFlowModelWrapper(model, extractor, tokenizer, label_list=[0, 1], max_seq_len=max_seq_length, clean_function=cleaning_function_bert)


    df = load_dataset_from_csv("../saved_models/fine_tuned/20201117_bert_model_imdb_reviews_exp_0/df_test.csv")

    texts = df["text"][400:448].tolist()
    true_labels = df["label"][400:448].tolist()

    tokens = tokenizer.tokenize(texts[0])
    print("Tokens: {}".format(tokens))
    word_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("Word Ids: {}".format(word_ids))

    #true_labels = None


    #embeddings = model_wrapper.extract_embedding(input_texts=texts, batch_size=32)

    #print(embeddings)

    #predictions = model_wrapper.predict(texts)

    #print(predictions)

    exp = explainer.LocalExplainer(model_wrapper, "20201117_bert_model_imdb_reviews_exp_0")

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



