import explainer
import bert_model_wrapper
import re
import os
import keras
import tensorflow as tf
import pandas as pd
from official.nlp.bert import tokenization

# RUN this the first time to load nltk models
# import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')


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

    max_seq_length = 256

    extractor = tf.keras.Model(inputs=model.inputs,
                               outputs=[model.get_layer("bert_layer").output])

    model.summary()

    return model, tokenizer, max_seq_length, extractor


if __name__ == "__main__":

    input_texts = ["You may fuck you think that azz!!!! abstract methods can't be implemented in the abstract base class. " 
                 "This impression is wrong: An abstract method can have an implementation in the abstract class!" 
                 " Even if they are implemented, designers of subclasses will be forced to override the implementation." 
                 " Like in other cases of normal inheritance, the abstract method can be invoked with super() call mechanism. " 
                 "This enables providing some basic functionality in the abstract method, which can be enriched by the subclass implementation."]

    model_dir = "saved_models/fine_tuned/20201128_bert_model_ag_news_subset_exp0"

    model, tokenizer, max_seq_length, extractor = load_model(model_dir)

    model_wrapper = bert_model_wrapper.BertModelWrapper(model, extractor, tokenizer, label_list=[0, 1, 2, 3], max_seq_len=max_seq_length, clean_function=cleaning_function_bert)

    #texts = ['''Bush, Lawmakers Discuss Social Security (AP). AP - President Bush sought support from congressional ''']

    #tokens = tokenizer.tokenize(texts[0])
    #print("Tokens: {}".format(tokens))
    #word_ids = tokenizer.convert_tokens_to_ids(tokens)
    #print("Word Ids: {}".format(word_ids))

    df = load_dataset_from_csv("saved_models/fine_tuned/20201128_bert_model_ag_news_subset_exp0/df_test.csv")

    texts = df["text"][100:105].tolist()
    true_labels = df["label"][100:105].tolist()

    #embeddings = model_wrapper.extract_embedding(input_texts=texts, batch_size=32)

    #print(embeddings)

    exp = explainer.LocalExplainer(model_wrapper, "20201128_bert_model_ag_news_subset_exp0")

    exp.fit_transform(input_texts=texts,
                      classes_of_interest=[-1]*len(texts),
                      expected_labels=true_labels,
                      flag_pos=True,
                      flag_sen=True,
                      flag_mlwe=True,
                      flag_combinations=True)

    #exp.fit(input_texts, [1])


