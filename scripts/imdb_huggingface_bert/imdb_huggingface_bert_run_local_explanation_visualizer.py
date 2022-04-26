from model_wrappers import bert_huggingface_model_wrapper
from transformers import AutoModelForSequenceClassification,AutoTokenizer,BertConfig
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

def get_model_from_dir(path):
    config = BertConfig.from_pretrained(path, output_hidden_states=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, config=config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    return model, tokenizer

if __name__ == "__main__":
    MODEL_PATH = "../../saved_models/fine_tuned/bert-imdb-huggingface"

    model, tokenizer = get_model_from_dir(MODEL_PATH)

    bert_model = bert_huggingface_model_wrapper.BertModelWrapper(model, tokenizer, clean_function=clean_text)

    print("End Main.")