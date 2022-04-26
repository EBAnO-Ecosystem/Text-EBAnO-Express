import visualizer
import explainer
import os
from utils import utils

LABEL_LIST = [0, 1]
LABEL_NAME_LIST = ["Unacceptable", "Acceptable"]

OUTPUT_FOLDER = "outputs"
USE_CASE_NAME = "ulmfit_model_cola_exp0"
GLOBAL_EXPLANATIONS_FOLDER = os.path.join(utils.get_project_root(), OUTPUT_FOLDER, USE_CASE_NAME, "global_explanations_experiments")
LOCAL_EXPLANATIONS_FOLDER = os.path.join(utils.get_project_root(), OUTPUT_FOLDER, USE_CASE_NAME, "local_explanations_experiments")


if __name__ == "__main__":

    LOCAL_EXPLANATION_EXPERIMENT_FOLDER = "20210401_111054_analysis"
    LOCAL_EXPLANATION_REPORT_FILENAME = "local_explanation_report_490.json"

    LOCAL_EXPLANATION_REPORT_PATH = os.path.join(LOCAL_EXPLANATIONS_FOLDER, LOCAL_EXPLANATION_EXPERIMENT_FOLDER,
                                                 "local_explanations", LOCAL_EXPLANATION_REPORT_FILENAME)

    # Instantiate the localExplanationReport class
    report = explainer.LocalExplanationReport()

    # Read the local_explanation_report from json file
    report.fit_local_explanation_report_from_json_file(LOCAL_EXPLANATION_REPORT_PATH)

    # Instantiate the localExplanationReportVisualizer class
    report_visualizer = visualizer.LocalExplanationReportVisualizer()

    # Fit the localExplanationReportVisualizer with the localExplanationReport
    report_visualizer.fit(report, label_names=LABEL_NAME_LIST)

    # Produce the HTML output from the localExplanationReport
    report_visualizer.visualize_report_as_html()

