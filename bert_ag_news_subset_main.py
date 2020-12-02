import explainer
import bert_model_wrapper
import re
import os
import keras
import tensorflow as tf

from official.nlp.bert import tokenization


def cleaning_function_bert(text):
    text = re.sub("@\S+", " ", text) # Remove Mentions
    text = re.sub("https*\S+", " ", text) # Remove URL
    text = re.sub("#\S+", " ", text) # Remove Hastags
    text = re.sub('&lt;/?[a-z]+&gt;', '', text) # Remove special Charaters
    text = re.sub('#39', ' ', text) # Remove special Charaters
    text = re.sub('<.*?>', '', text) # Remove html
    text = re.sub(' +', ' ', text) # Merge multiple blank spaces

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


if __name__ == "__main__":

    input_texts = ["You may fuck you think that azz!!!! abstract methods can't be implemented in the abstract base class. " 
                 "This impression is wrong: An abstract method can have an implementation in the abstract class!" 
                 " Even if they are implemented, designers of subclasses will be forced to override the implementation." 
                 " Like in other cases of normal inheritance, the abstract method can be invoked with super() call mechanism. " 
                 "This enables providing some basic functionality in the abstract method, which can be enriched by the subclass implementation."]

    model_dir = "saved_models/fine_tuned/20201128_bert_model_ag_news_subset_exp0"

    model, tokenizer, max_seq_length, extractor = load_model(model_dir)

    model_wrapper = bert_model_wrapper.BertModelWrapper(model, extractor, tokenizer, label_list=[0, 1, 2, 3], max_seq_len=max_seq_length, clean_function=cleaning_function_bert)

    texts =["This film is very awful. I have never seen such a bad movie.", "I really love this film. It is probably one of my favourite movie of ever.",
            "The book is very nice. The film instead is not as good as the book. I expected more for this movie."]

    texts = ['''Bush, Lawmakers Discuss Social Security (AP). AP - President Bush sought support from congressional 
      leaders of both parties Monday for his aggressive proposal to overhaul Social Security during his second term.''','''Bush, Lawmakers Discuss Social Security (AP). AP - President Bush sought support from congressional 
      leaders of both parties Monday for his aggressive proposal to overhaul Social Security during his second term.''','''Bush, Lawmakers Discuss Social Security (AP). AP - President Bush sought support from congressional 
      leaders of both parties Monday for his aggressive proposal to overhaul Social Security during his second term.''','''Bush, Lawmakers Discuss Social Security (AP). AP - President Bush sought support from congressional 
      leaders of both parties Monday for his aggressive proposal to overhaul Social Security during his second term.''','''Bush, Lawmakers Discuss Social Security (AP). AP - President Bush sought support from congressional 
      leaders of both parties Monday for his aggressive proposal to overhaul Social Security during his second term.''','''Bush, Lawmakers Discuss Social Security (AP). AP - President Bush sought support from congressional 
      leaders of both parties Monday for his aggressive proposal to overhaul Social Security during his second term.''','''Bush, Lawmakers Discuss Social Security (AP). AP - President Bush sought support from congressional 
      leaders of both parties Monday for his aggressive proposal to overhaul Social Security during his second term.''','''Bush, Lawmakers Discuss Social Security (AP). AP - President Bush sought support from congressional 
      leaders of both parties Monday for his aggressive proposal to overhaul Social Security during his second term.''','''Bush, Lawmakers Discuss Social Security (AP). AP - President Bush sought support from congressional 
      leaders of both parties Monday for his aggressive proposal to overhaul Social Security during his second term.''','''Bush, Lawmakers Discuss Social Security (AP). AP - President Bush sought support from congressional 
      leaders of both parties Monday for his aggressive proposal to overhaul Social Security during his second term.''']

    #texts = ['''Bush, Lawmakers Discuss Social Security - President Bush sought support from congressional
          #leaders of both Monday his aggressive proposal to overhaul Social Security during his second term.''']

    tokens = tokenizer.tokenize(texts[0])
    print("Tokens: {}".format(tokens))
    word_ids = tokenizer.convert_tokens_to_ids(tokens)
    print("Word Ids: {}".format(word_ids))

    #embeddings = model_wrapper.extract_embedding(input_texts=texts, batch_size=32)

    #print(embeddings)

    exp = explainer.ExplainerClass(model_wrapper, "20201128_bert_model_ag_news_subset_exp0")

    exp.fit_transform(texts, [-1]*len(texts), True, True, True, True)

    #exp.fit(input_texts, [1])


