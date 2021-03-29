import visualizer
import explainer
import os
from utils import utils

LABEL_LIST = [0, 1, 2, 3]
LABEL_NAME_LIST = ["Negative", "Positive"]

OUTPUT_FOLDER = "outputs"
USE_CASE_NAME = "20201117_bert_model_imdb_reviews_exp_0"
GLOBAL_EXPLANATIONS_FOLDER = os.path.join(utils.get_project_root(), OUTPUT_FOLDER, USE_CASE_NAME, "global_explanations_experiments")
LOCAL_EXPLANATIONS_FOLDER = os.path.join(utils.get_project_root(), OUTPUT_FOLDER, USE_CASE_NAME, "local_explanations_experiments")

