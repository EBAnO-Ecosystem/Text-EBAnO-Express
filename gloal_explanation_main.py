import explainer


if __name__ == "__main__":
    report_folder_path = "/Users/salvatore/PycharmProjects/T-EBAnO-Express/outputs/20201117_bert_model_imdb_reviews_exp_0/" \
                         "local_explanations_experiments/20201204_155209/local_explanations"

    global_explanation_report_path = "/Users/salvatore/PycharmProjects/T-EBAnO-Express/outputs/20201117_bert_model_imdb_reviews_exp_0/global_explanations_experiments"

    global_explainer = explainer.GlobalExplainer(label_list=[0, 1], label_names=["Negative, Positive"])

    global_explainer.fit_from_folder(report_folder_path)

    global_explanation_report = global_explainer.transform()

    global_explanation_report.save_global_explanation_report(global_explanation_report_path)

