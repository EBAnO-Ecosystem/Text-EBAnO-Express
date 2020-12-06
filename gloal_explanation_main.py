import explainer


if __name__ == "__main__":
    report_folder_path = "/Users/salvatore/PycharmProjects/T-EBAnO-Express/outputs/20201117_bert_model_imdb_reviews_exp_0/local_explanations_experiments/20201205_173717/local_explanations"

    global_explanation_report_path = "/Users/salvatore/PycharmProjects/T-EBAnO-Express/outputs/20201117_bert_model_imdb_reviews_exp_0/global_explanations_experiments"
    #global_explanation_report_path = "/Users/salvatore/PycharmProjects/T-EBAnO-Express/outputs/20201128_bert_model_ag_news_subset_exp0/global_explanations_experiments"


    label_list = [0,1]
    label_names = ["negative","positive"]

    #label_names = ["sci", "bus", "rel", "bo"]
    #label_list = [0,1,2,3]

    global_explainer = explainer.GlobalExplainer(label_list=label_list, label_names=label_names)

    global_explainer.fit_from_folder(report_folder_path)

    global_explanation_report = global_explainer.transform(feature_type="MLWE")

    global_explanation_report.save_global_explanation_report(global_explanation_report_path)

    most_informative_le_bush = global_explanation_report.get_most_informative_local_explanations_by_token("bush")

    print(most_informative_le_bush)

    print("\n")