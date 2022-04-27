from model_wrappers import bert_huggingface_model_wrapper
from transformers import AutoModelForSequenceClassification,AutoTokenizer,BertConfig
from datasets import load_dataset

import explainer
import re


def clean_text(text):
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

def get_model_from_dir(path, tokenizer_string="bert-base-uncased"):
    config = BertConfig.from_pretrained(path, output_hidden_states=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_string)

    return model, tokenizer

if __name__ == "__main__":
    USE_CASE_NAME = "imdb_huggingface_bert"
    MODEL_PATH = "../../saved_models/fine_tuned/bert-imdb-huggingface"
    TOKENIZER_STRING = "bert-base-uncased"
    DATASET_NAME = "imdb"
    N = 1

    model, tokenizer = get_model_from_dir(MODEL_PATH, TOKENIZER_STRING)

    raw_datasets = load_dataset(DATASET_NAME)

    test_texts = raw_datasets["test"]["text"]
    test_labels = raw_datasets["test"]["label"]

    texts = test_texts[:N]
    true_labels = test_labels[:N]

    bert_model_wrapper = bert_huggingface_model_wrapper.BertModelWrapper(model, tokenizer, clean_function=clean_text)

    coi = -1

    # Instantiate the LocalExplainer class for the current mdoel
    exp = explainer.LocalExplainer(bert_model_wrapper, model_name=USE_CASE_NAME)

    exp.fit_transform(input_texts=texts,
                      classes_of_interest=[coi] * len(texts),
                      expected_labels=true_labels,
                      flag_pos=False,
                      flag_sen=False,
                      flag_mlwe=True,
                      flag_combinations=False,
                      flag_rnd=False)

    print("End Main.")