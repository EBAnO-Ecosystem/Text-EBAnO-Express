import explainer


if __name__ == "__main__":
    report_folder_path = "/Users/salvatore/PycharmProjects/T-EBAnO-Express/outputs/20201117_bert_model_imdb_reviews_exp_0/" \
                         "local_explanations_experiments/20201204_105147/local_explanations"

    global_explainer = explainer.GlobalExplainer(label_list=[0, 1], label_names=["Negative, Positive"])

    global_explainer.fit_from_folder(report_folder_path)

    global_explainer.transform()

