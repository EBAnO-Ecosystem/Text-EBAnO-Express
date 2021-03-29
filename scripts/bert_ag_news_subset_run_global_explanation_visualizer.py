import visualizer
import explainer
import os
from utils import utils

LABEL_LIST = [0, 1, 2, 3]
LABEL_NAME_LIST = ["World", "Sport", "Business", "Sci/Tech"]

OUTPUT_FOLDER = "outputs"
USE_CASE_NAME = "20201128_bert_model_ag_news_subset_exp0"
GLOBAL_EXPLANATIONS_FOLDER = os.path.join(utils.get_project_root(), OUTPUT_FOLDER, USE_CASE_NAME, "global_explanations_experiments")
LOCAL_EXPLANATIONS_FOLDER = os.path.join(utils.get_project_root(), OUTPUT_FOLDER, USE_CASE_NAME, "local_explanations_experiments")


if __name__ == "__main__":
    report_folder_path = "../outputs/20201128_bert_model_ag_news_subset_exp0/local_explanations_experiments/20201215_173708/local_explanations"

    #global_explanation_report_path = "/Users/salvatore/PycharmProjects/T-EBAnO-Express/outputs/20201117_bert_model_imdb_reviews_exp_0/global_explanations_experiments"
    global_explanation_report_path = "/outputs/20201128_bert_model_ag_news_subset_exp0/global_explanations_experiments"

    LOCAL_EXPLANATION_EXPERIMENT_FOLDER = "20210318_101319"
    LOCAL_EXPLANATION_REPORT_FILENAME = "local_explanation_report_0.json"

    LOCAL_EXPLANATION_REPORT_PATH = os.path.join(LOCAL_EXPLANATIONS_FOLDER, LOCAL_EXPLANATION_EXPERIMENT_FOLDER,
                                                 "local_explanations", LOCAL_EXPLANATION_REPORT_FILENAME)

    global_explainer = explainer.GlobalExplainer(label_list=LABEL_LIST, label_names=LABEL_NAME_LIST)

    global_explainer.fit_from_folder(report_folder_path)

    global_explanation_report = global_explainer.transform(feature_type="MLWE")

    global_explanation_report.save_global_explanation_report(global_explanation_report_path)

    most_informative_le_bush = global_explanation_report.get_most_informative_local_explanations_by_token("afp")

    print(most_informative_le_bush)

    print("\n")