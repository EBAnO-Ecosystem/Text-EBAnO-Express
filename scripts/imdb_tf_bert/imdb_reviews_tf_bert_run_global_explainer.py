import explainer


if __name__ == "__main__":
    report_folder_path = ""

    global_explanation_report_path = ""

    global_explainer = explainer.GlobalExplainer(label_list=[0, 1], label_names=["Negative, Positive"])

    global_explainer.fit_from_folder(report_folder_path)

    global_explanation_report = global_explainer.transform()

    global_explanation_report.save_global_explanation_report(global_explanation_report_path)