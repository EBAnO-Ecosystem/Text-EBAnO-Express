import tensorflow as tf
import pickle
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def load_saved_model(model_filepath):
    """ Load keras saved model from disk given the filepath.

    Args:
        model_filepath (str): String containing the model filepath
    Returns:
        model (tf.keras.models) keras Model loaded from disk
    """
    return tf.keras.models.load_model(model_filepath)


def load_tokenizer(tokenizer_filepath):
    """ Load keras saved model from disk given the filepath.

    Args:
        tokenizer_filepath (str): String containing the tokenizer filepath
    Returns:
        tokenizer (tf.keras.preprocessing.text.Tokenizer) keras Tokenizer loaded from disk
    """
    with open(tokenizer_filepath, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer
