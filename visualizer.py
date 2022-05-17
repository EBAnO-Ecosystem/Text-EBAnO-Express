import explainer
from config_files import local_explanation_report_template
import os
from utils import utils

def html_str_to_file(text, filename):
    """Write a file with the given name and the given text."""
    output = open(os.path.join(utils.get_project_root(), "outputs","tmp",filename), "w")
    output.write(text)
    output.close()
    return

def browseLocal(webpageText, filename='tempLocalExplanationReport.html'):
    """ Start your webbrowser on a local file containing the text with given filename. """
    import webbrowser, os.path
    html_str_to_file(webpageText, filename)
    webbrowser.open("file:///" + os.path.abspath(os.path.join(utils.get_project_root(), "outputs","tmp",filename)))
    return


class LocalExplanationVisualizer:
    def __init__(self, local_explanation, label_names):
        self.local_explanation = local_explanation
        self.label_names = label_names
        return


class LocalExplanationReportVisualizer:
    def __init__(self):
        self.local_explanation_report = None
        return

    def fit(self, local_explanation_report, label_names=["negative", "positive"]):
        self.local_explanation_report = local_explanation_report
        self.label_names = label_names
        return

    def html_highlight_feature_into_text(self, input_positions_tokens, local_explanation, r=0, g=255, b=255, a=1):
        html_string = """<h4>Feature {} <br> {} - {}</h4> <div class="featureBoxed">""".format(local_explanation.perturbation.feature.feature_id,
                                                                                           local_explanation.perturbation.feature.feature_type,
                                                                                           local_explanation.perturbation.feature.description)

        feature_color = "background-color:rgba({}, {}, {}, {})".format(r, g, b, a)
        for position in range(len(input_positions_tokens)):
            if str(position) in local_explanation.perturbation.feature.positions_tokens:
                html_string = html_string + '<span style="{}"><b>{}</b></span> '.format(feature_color,
                                                                                        input_positions_tokens[
                                                                                            position])
            else:
                html_string = html_string + '<span><b>{}</b></span> '.format(input_positions_tokens[position])

        html_string = html_string + """</div>
                                        <div class="featureBoxed"><b>{}</b></div>""".format(local_explanation.perturbation.perturbed_text)

        if local_explanation.numerical_explanation.nPIR_original_top_class >= 0.2:
            influential_string = "positive_influential_color"
        else:
            if local_explanation.numerical_explanation.nPIR_original_top_class <= -0.2:
                influential_string = "negative_influential_color"
            else:
                influential_string = "neutral_influential_color"

        change_probabilities = [local_explanation.perturbed_probabilities[i] - self.local_explanation_report.original_probabilities[i]
                                for i in range(len(self.local_explanation_report.original_probabilities))]

        change_probabilities_string = "[ "
        for p in change_probabilities:
            if (p >= 0):
                change_probabilities_string = change_probabilities_string + """<span id="positive_influential_color">+""" + str(round(p,3))
            else:
                change_probabilities_string = change_probabilities_string + """<span id="negative_influential_color">""" + str(round(p, 3))
            change_probabilities_string = change_probabilities_string + """</span>  ,  """

        change_probabilities_string = change_probabilities_string + " ]"


        html_string = html_string + """
                            <div id="perturbation_info">
                                <table id="table_perturbation">
                                    <tr>
                                        <th>nPIR</th>
                                        <th>Perturbed Probabilities</th>
                                        <th>Perturbed Label</th>
                                        <th>Perturbed Label Name</th>
                                    </tr>
                                    <tr>
                                        <th id="{}">{}</th>
                                        <th>{} {}</th>
                                        <th>{}</th>
                                        <th>{}</th>
                                    </tr>
                                </table>
                            </div>""".format(influential_string,
                                             round(local_explanation.numerical_explanation.nPIR_original_top_class, 3),
                                             [round(p_p, 3) for p_p in local_explanation.perturbed_probabilities],
                                             change_probabilities_string,
                                             local_explanation.perturbed_top_class,
                                             self.label_names[local_explanation.perturbed_top_class])

        return html_string

    def get_html_string_summary_feature_type(self, feature_type=any(["MLWE", "POS", "SEN","RND"])):
        # get features of `feature_type` without combinations
        filtered_local_explanations = self.local_explanation_report.get_filtered_local_explanations(feature_type, [1])

        positions_tokens_score = {}  # Dictionary with `position` as key and as value the tuple `(token, nPIR)`

        for le in filtered_local_explanations:
            for position, token in le.perturbation.feature.positions_tokens.items():
                positions_tokens_score[position] = (token, round(le.numerical_explanation.nPIR_original_top_class, 4))

        html_string = ""
        for position in sorted(positions_tokens_score.keys(), key=lambda k: int(k)):
            token_score = positions_tokens_score[position]
            if token_score[1] >= 0:
                positive_color = "background-color:rgba(124, 252, 0, {})".format(token_score[1])
                html_string = html_string + '<span style="{}"><b>{}</b></span> '.format(positive_color, token_score[0])
            else:
                negative_color = "background-color:rgba(255, 99, 71, {})".format(token_score[1])
                html_string = html_string + '<span style="{}"><b>{}</b></span> '.format(negative_color, token_score[0])

        return html_string

    def visualize_report_as_html(self):
        html_mlwe_summary = self.get_html_string_summary_feature_type("MLWE")
        html_pos_summary = self.get_html_string_summary_feature_type("POS")
        html_sen_summary = self.get_html_string_summary_feature_type("SEN")
        html_rnd_summary = self.get_html_string_summary_feature_type("RND")

        html_mlwe_explanations = ""
        for l_e in sorted(self.local_explanation_report.get_filtered_local_explanations(feature_type_list="MLWE", combination_list=[1, 2]),
                         key=lambda local_explanation: local_explanation.numerical_explanation.nPIR_original_top_class, reverse=True):
                current_exp = self.html_highlight_feature_into_text(self.local_explanation_report.positions_tokens, l_e)
                html_mlwe_explanations = html_mlwe_explanations + "<hr>" + current_exp

        html_pos_explanations = ""
        for l_e in sorted(self.local_explanation_report.get_filtered_local_explanations(feature_type_list="POS", combination_list=[1, 2]),
                         key=lambda local_explanation: local_explanation.numerical_explanation.nPIR_original_top_class, reverse=True):
                current_exp = self.html_highlight_feature_into_text(self.local_explanation_report.positions_tokens, l_e)
                html_pos_explanations = html_pos_explanations + "<hr>" + current_exp

        html_sen_explanations = ""
        for l_e in sorted(self.local_explanation_report.get_filtered_local_explanations(feature_type_list="SEN", combination_list=[1, 2]),
                         key=lambda local_explanation: local_explanation.numerical_explanation.nPIR_original_top_class, reverse=True):
                current_exp = self.html_highlight_feature_into_text(self.local_explanation_report.positions_tokens, l_e)
                html_sen_explanations = html_sen_explanations + "<hr>" + current_exp

        html_rnd_explanations = ""
        for l_e in sorted(self.local_explanation_report.get_filtered_local_explanations(feature_type_list="RND", combination_list=[1, 2]),
                         key=lambda local_explanation: local_explanation.numerical_explanation.nPIR_original_top_class, reverse=True):
                current_exp = self.html_highlight_feature_into_text(self.local_explanation_report.positions_tokens, l_e)
                html_rnd_explanations = html_rnd_explanations + "<hr>" + current_exp

        contents = local_explanation_report_template.localExplanationReportTemplate.format(raw_text=self.local_explanation_report.raw_text,
                                                                                           clean_text=self.local_explanation_report.cleaned_text,
                                                                                           pre_text=self.local_explanation_report.preprocessed_text,
                                                                                           html_mlwe_summary=html_mlwe_summary,
                                                                                           html_pos_summary=html_pos_summary,
                                                                                           html_sen_summary=html_sen_summary,
                                                                                           html_rnd_summary=html_rnd_summary,
                                                                                           mlwe_local_explanations=html_mlwe_explanations,
                                                                                           pos_local_explanations=html_pos_explanations,
                                                                                           sen_local_explanations=html_sen_explanations,
                                                                                           rnd_local_explanations=html_rnd_explanations,
                                                                                           original_probabilities=[round(o_p,3) for o_p in self.local_explanation_report.original_probabilities],
                                                                                           original_label=self.local_explanation_report.original_label,
                                                                                           label_name=self.label_names[
                                                                                               self.local_explanation_report.original_label
                                                                                           ],
                                                                                           )



        browseLocal(contents)
        return


class GlobalExplanationReportVisualizer:

    def __init__(self):
        self.global_explanation_report = None

    def fit(self, global_explanation_report):
        self.global_explanation_report = global_explanation_report
        return
