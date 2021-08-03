from model_wrappers import keras_glove_model_wrapper
import explainer
from keras.preprocessing.sequence import pad_sequences
from utils import utils
import numpy as np
import pandas as pd
import re
import os
import keras
import pickle
import tqdm

MODEL_DIR = os.path.join("saved_models", "fine_tuned")
USE_CASE_NAME = "lstm_model_toxic_comment_exp0"
FINE_TUNED_MODEL_DIR = os.path.join(utils.get_project_root(), MODEL_DIR, USE_CASE_NAME)

GLOVE_DIR = os.path.join(utils.get_project_root(),"saved_models", "pre_trained", "GloVe")
GLOVE_MODEL = "glove.840B.300d.txt"
GLOVE_MODEL_PATH = os.path.join(GLOVE_DIR, GLOVE_MODEL)

LABEL_LIST = [0, 1]
MAX_SEQ_LENGTH = 300
PAD_TYPE = 'post'
TRUNC_TYPE = 'post'

DATASET_NAME = "civil_comments"
DF_DATASET_PATH = os.path.join(utils.get_project_root(), "datasets", DATASET_NAME, "dataset_custom.csv")


def cleaning_function_lstm(text):
    RE_PATTERNS = {
        ' fuck':
            [
                '(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*',
                '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)',
                ' f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k', 'f u u c',
                '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\*',
                'feck ', ' fux ', 'f\*\*',
                'f\.u\.', 'f###', ' fu ', 'f@ck', 'f u c k', 'f uck', 'f ck', 'fuk', 'fk'
            ],
        ' ass ':
            [
                '[^a-z]ass ', '[^a-z]azz ', 'arrse', ' arse ', '@\$\$', '[^a-z]anus', ' a\*s\*s', '[^a-z]ass[^a-z ]',
                'a[@#\$%\^&\*][@#\$%\^&\*]', '[^a-z]anal ', 'a s s'
            ],
        ' asshole ':
            [
                ' a[s|z]*wipe', 'a[s|z]*[w]*h[o|0]+[l]*e', '@\$\$hole'
            ],
        ' bitch ':
            [
                'b[w]*i[t]*ch', 'b!tch',
                'bi\+ch', 'b!\+ch', '(b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)',
                'biatch', 'bi\*\*h', 'bytch', 'b i t c h'
            ],
        ' bastard ':
            [
                'ba[s|z]+t[e|a]+rd'
            ],
        ' trans gender':
            [
                'transgender'
            ],
        ' gay ':
            [
                'gay', 'g4y'
            ],
        ' cock ':
            [
                '[^a-z]cock', 'c0ck', '[^a-z]cok ', 'c0k', '[^a-z]cok[^aeiou]', ' cawk',
                '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'
            ],
        ' dick ':
            [
                ' dick[^aeiou]', 'deek', 'd i c k', 'dik'
            ],
        ' suck ':
            [
                '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'suck', '5uck', 's u c k'
            ],
        ' cunt ':
            [
                'cunt', 'c u n t'
            ],
        ' bullshit ':
            [
                'bullsh\*t', 'bull\$hit'
            ],
        ' idiot ':
            [
                'i[d]+io[t]+', '(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)', 'i d i o t'

            ],
        ' dumb ':
            [
                '(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'
            ],
        ' shit ':
            [
                'shitty', '(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)', 'shite', '\$hit', 's h i t'
            ],
        ' shit hole ':
            [
                'shythole'
            ],
        ' retard ':
            [
                'returd', 'retad', 'retard', 'wiktard', 'wikitud'
            ],
        ' rape ':
            [
                ' raped'
            ],
        ' dumb ass':
            [
                'dumbass', 'dubass'
            ],
        ' ass head':
            [
                'butthead'
            ],
        ' sex ':
            [
                's3x'
            ],
        ' nigger ':
            [
                'nigger', ' nigr ', 'negrito', 'niguh', 'n3gr', 'n i g g e r'
            ],
        ' nigga ':
            [
                'niga', 'ni[g]+a', ' nigg[a]+'
            ],
        ' shut the fuck up':
            [
                'stfu'
            ],
        ' pussy ':
            [
                'pussy[^c]', 'pusy', 'pussi[^l]', 'pusses'
            ],
        ' faggot ':
            [
                'faggot', ' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit',
                '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', 'fau[g]+ot', 'fae[g]+ot',
            ],
        ' motherfucker':
            [
                ' motha ', ' motha f', ' mother f', 'motherucker',
            ],
        ' whore ':
            [
                'w h o r e'
            ],
    }

    def clean_text(t):
        t = t.lower()
        t = re.sub(r"what's", "what is ", t)
        t = re.sub(r"\'s", " ", t)
        t = re.sub(r"\'ve", " have ", t)
        t = re.sub(r"can't", "cannot ", t)
        t = re.sub(r"n't", " not ", t)
        t = re.sub(r"i'm", "i am ", t)
        t = re.sub(r"\'re", " are ", t)
        t = re.sub(r"\'d", " would ", t)
        t = re.sub(r"\'ll", " will ", t)
        t = re.sub(r"\'scuse", " excuse ", t)
        t = re.sub('\W', ' ', t)
        t = re.sub('\s+', ' ', t)
        t = t.strip(' ')
        return t


    text = clean_text(text)

    for target, patterns in RE_PATTERNS.items():
        for pat in patterns:
            text = re.sub(pat, target, text)

    return text

def load_dataset_from_csv(dataset_path):
    df = pd.read_csv(dataset_path)
    return df


def preprocessing_function(text, tokenizer):
    sequences = tokenizer.texts_to_sequences(text)

    input_padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=300)

    return input_padded

def load_model(saved_model_dir=FINE_TUNED_MODEL_DIR, model_name="model.h5"):
    model = keras.models.load_model(os.path.join(saved_model_dir, model_name))
    return model

def load_tokenizer(saved_model_dir=FINE_TUNED_MODEL_DIR, tokenizer_name="tokenizer.pickle"):
    with open(os.path.join(saved_model_dir, tokenizer_name), 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def load_glove_word_embedding(glove_path=GLOVE_MODEL_PATH):
    glove_embedding = {}
    f = open(glove_path,'r')
    for line in f:
        value = line.split(' ')
        word = value[0]
        coef = np.array(value[1:], dtype='float32')
        glove_embedding[word] = coef

    return glove_embedding


if __name__ == "__main__":

    df_dataset = load_dataset_from_csv(DF_DATASET_PATH)

    df_toxic = df_dataset.loc[df_dataset['target'] >= 0.5]

    texts = df_toxic["comment_text"][:100].tolist()
    true_labels = df_toxic["target"][:100].tolist()

    tokenizer = load_tokenizer()
    model = load_model()
    glove_embedding = load_glove_word_embedding()

    model_wrapper = keras_glove_model_wrapper.KerasGloveModelWrapper(model, glove_embedding, tokenizer,
                                                                     label_list=LABEL_LIST, max_seq_len=300, seq_padding='post',
                                                                     seq_truncating='post', clean_function=cleaning_function_lstm)

    # Instantiate the LocalExplainer class for the current mdoel
    exp = explainer.LocalExplainer(model_wrapper, model_name=USE_CASE_NAME)

    exp.fit_transform(input_texts=texts,
                      classes_of_interest=[-1]*len(texts),
                      expected_labels=true_labels,
                      flag_pos=True,
                      flag_sen=True,
                      flag_mlwe=True,
                      flag_rnd=True,
                      flag_combinations=True)

    #exp.fit(input_texts, [1])


