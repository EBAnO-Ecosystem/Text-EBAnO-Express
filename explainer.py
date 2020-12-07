import feature_extraction as fe
import perturbation as pe
import local_explanation as le
from typing import List
from nltk.stem import WordNetLemmatizer

import yaml
import os
import time
import datetime
import json


class LocalExplainer:
    """ Local Explainer Class.

        Performs a set of local explanations for a given Black-Box model and a set of input texts.

         Attributes:
            model_wrapper (:obj:ModelWrapperInterface)  Instance of a real class implementing the ModelWrapperInterface
            model_name
        """
    def __init__(self, model_wrapper, model_name):
        # Instantiate the model wrapper
        self.model_wrapper = model_wrapper
        self.model_name = model_name

        # Attributes assigned with fit method
        self.raw_texts = None
        self.cleaned_texts = None
        self.preprocessed_texts = None
        self.sequences = None
        self.tokens_list = None
        self.classes_of_interest = None
        self.word_index = None
        self.index_word = None
        self.expected_labels = None
        self.input_names = None

    def fit(self, input_texts:  List[str], classes_of_interest: List[int], expected_labels: List[int] = None, input_names: List[any] = None):
        """ Fits the explainer with a list of input texts to be explained.

        The number of elements in `input_texts` and `classes_of_interests` must be equal.
        Each element of the list `input_texts` is a string containing a text to be locally explained with the model_wrapper
        passed in the constructor. Each element of `classes_of_interest` is an integer containing the label for which perform
        the explanation. The value `-1` perform the explanation for the most likely label predicted by the model for the current
        input text. `input_names` is an optional parameter that, if passed, need to match the length of `input_texts` and `classes_of_interest`.
        If passed, the input_name is used to compose the `local_explanation_report_name` as: `local_explanation_report_{input_name}_{report_id}.json`,
        otherwise it will be: `local_explanation_report_{report_id}.json`.

        Args:
            input_texts (List[str]): Fits the explainer with a list of input texts to be locally explained. For each input will be produced
                                     a local explanation report.
            classes_of_interest (List[int]): Fits the explainer with the list of classes of interest (-1 implies that the
                                             class of interest is the label with higher probabilities in the original prediction).
            expected_labels (List[int])[Optional]: Fits the explainer with the list of true expected labels (one for each input text).
            input_names (List[str])[Optional]: Fits the explainer with a list of names (one for each input). If passed, this information will
                                               be used to compose the local explanation report name.
        Raises:
            ValueError: if `input_texts` is not a list of strings
            ValueError: if `classes_of_interest` is not a list of integers
            ValueError: if `input_names` is not a list of integers (if passed as parameter)
            ValueError: if `len(input_texts)` and `len(classes_of_interest)` are not equal
        """
        self.__check_fit_parameters(input_texts, classes_of_interest, expected_labels, input_names)  # Check parameters of fit method
        if input_names is None:
            self.input_names = [None]*len(input_texts)
        else:
            self.input_names = input_names

        if expected_labels is None:
            self.expected_labels = [None]*len(input_texts)
        else:
            self.expected_labels = expected_labels

        self.raw_texts = input_texts  # List[str] containing the original raw input texts
        self.classes_of_interest = classes_of_interest  # List[int] containing the class_of_interest for the explanation of each input
        self.cleaned_texts = [self.model_wrapper.clean_function(text) for text in input_texts]  # List[str] Clean each input with the clean_function specified in the model_wrapper
        self.sequences = self.model_wrapper.texts_to_sequences(self.cleaned_texts)  # List[List[int]] list of sequences ids (one sequence of ids for each input text)
        self.tokens_list = self.model_wrapper.texts_to_tokens(self.cleaned_texts)  # List[List[str]] list of sequences tokens (one sequence of tokens for each input text)
        self.preprocessed_texts = self.model_wrapper.sequences_to_texts(self.sequences)  # List[str] list of texts by joining each sequence of tokens
        return

    def transform(self, flag_pos: bool = True, flag_sen: bool = True, flag_mlwe: bool = True, flag_combinations: bool = True, output_folder: str = None):
        """ Create a local explanation report for each input text passed in the fit method.

        For each text passed in the fit method, it perform the local explanation for the relative class of interest and
        save one explanation report of each input text separately. The types of feature extraction methods performed are
        based on the boolean parameters passed.

        Args:
            flag_pos (bool): True if want perform Part-of-Speech Feature Extraction, False otherwise.
            flag_sen (bool): True if want perform Sentence Feature Extraction, False otherwise.
            flag_mlwe (bool): True if want perform Multi-Layer Word Embedding Feature Extraction, False otherwise.
            flag_combinations (bool): True if want perform (pairwise) combinations of features, False otherwise.
            output_folder (str)[Optional]: folder where save the outputs, the default folder is specified in the config.yaml file.
        Raises:
            ValueError: (flag_pos or flag_sen or flag_mlwe o flag_combinations) is not bool
        """
        self.__check_transform_parameters(flag_pos, flag_sen, flag_mlwe, flag_combinations)

        # Read configuration from configuration file
        with open(r'config_files/config.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

        # Create the output folders
        base_experiment_folder, local_explanations_folder = self.__prepare_output_folder(output_folder, config, timestamp)

        # Extract embedding tensor of each input at once
        if flag_mlwe is True:
            print("INFO: Extracting embedding of all input texts.")
            embedding_tensors = self.model_wrapper.extract_embedding(input_texts=self.cleaned_texts, batch_size=32)
            print("INFO: Embedding extracted successfully.")
        else:
            embedding_tensors = [None]*len(self.raw_texts)  # If `flag_mlwe` is False, then the embedding is not needed

        input_id = 0
        # Loop over each input text to perform the explanation
        for raw_text, cleaned_text, preprocessed_text, tokens, class_of_interest, embedding_tensor, expected_label, input_name in \
                zip(self.raw_texts, self.cleaned_texts, self.preprocessed_texts, self.tokens_list, self.classes_of_interest, embedding_tensors, self.expected_labels, self.input_names):

            print("INFO: Explaining text {}/{} ".format(input_id+1, len(self.raw_texts)))
            self.__perform_local_explanation_single_input_text(input_id,  # Will fill the report id number
                                                               raw_text,
                                                               cleaned_text,
                                                               preprocessed_text,
                                                               tokens,
                                                               class_of_interest,
                                                               embedding_tensor,
                                                               flag_pos, flag_sen, flag_mlwe, flag_combinations,
                                                               local_explanations_folder,
                                                               expected_label,
                                                               input_name)
            input_id += 1

        return

    def fit_transform(self, input_texts, classes_of_interest, expected_labels, flag_pos, flag_sen, flag_mlwe, flag_combinations):
        """ Fits the explainer with input texts and perform transform methods to create the local explanation reports.

        """
        self.fit(input_texts, classes_of_interest, expected_labels)

        self.transform(flag_pos, flag_sen, flag_mlwe, flag_combinations)
        return

    def __perform_local_explanation_single_input_text(self, input_id, raw_text, cleaned_text, preprocessed_text, tokens,
                                                      class_of_interest, embedding_tensor,
                                                      flag_pos, flag_sen, flag_mlwe, flag_combinations, local_explanations_folder, expected_label, input_name):
        """ Performs and saves the local explanation for a single input text.

        Given a single input and a class of interest, it performs the explanation process:
            1) Extract Features -> List[:obj:Feature]
            2) Apply a Perturbation to each Feature -> List[:obj:Perturbation]
            3) Create a Local Explanation for each Perturbation -> List[:obj:LocalExplanation]
        """

        # Start timestamp of the single explanation (relative to one input text)
        explanation_report_start_time = time.time()

        # Instantiate the FeaturesExtractionManager
        features_extraction_manager = fe.FeaturesExtractionManager(raw_text,  # Raw version of the current input text
                                                                   cleaned_text,  # Clean version of the current input text
                                                                   preprocessed_text,  # Preprocessed version of the current input text
                                                                   tokens,  # List of tokens of the current input text
                                                                   class_of_interest,  # Class of interest for the current input
                                                                   embedding_tensor,  # Embedding tensor of shape (n_tokens, n_features) of the current input text
                                                                   self.model_wrapper,
                                                                   flag_pos=flag_pos,  # True if want apply POS feature extraction
                                                                   flag_sen=flag_sen,  # True if want apply SEN feature extraction
                                                                   flag_mlwe=flag_mlwe,  # True if want apply MLWE feature extraction
                                                                   flag_combinations=flag_combinations)  # True for apply pairwise combination of features

        print("\tINFO: Feature Extraction Phase")
        # Extract the features from the input text
        features_extraction_manager.execute_feature_extraction_phase()

        # Get the list of all the features extracted -> List[Feature]
        features = features_extraction_manager.get_all_features()

        # Instantiate the PerturbationManager
        perturbation_manager = pe.PerturbationManager(preprocessed_text,  # Preprocessed version of the current input text
                                                      tokens,  # List of tokens of the current input text
                                                      self.model_wrapper,
                                                      features,  # List of features to which apply the perturbation
                                                      flag_removal=True)  # True if want apply Removal perturbation

        print("\tINFO: Perturbation Phase")
        # Perturb the features
        perturbation_manager.execute_perturbation_phase()

        # Get the list of all the perturbations produced -> List[Perturbation]
        perturbations = perturbation_manager.get_all_perturbations()

        # Instantiate the LocalExplanationManager
        local_explanation_manager = le.LocalExplanationManager(preprocessed_text,  # Preprocessed version of the current input text
                                                               self.model_wrapper,
                                                               class_of_interest,  # Class of interest for the current input
                                                               perturbations)  # List of Perturbations to which produce the local explanation

        print("\tINFO: Local Explanation Phase")
        # Produce the local explanation for each perturbation applied
        local_explanations, original_probabilities, original_label = local_explanation_manager.execute_local_explanation_phase()

        # End timestamp of the single explanation (relative to one input text)
        explanation_report_end_time = time.time()
        explanation_report_execution_time = explanation_report_end_time - explanation_report_start_time

        # Create an instance of the LocalExplanationReport class
        local_explanation_report = LocalExplanationReport()

        # Fits the local explanations in the LocalExplanationReport
        local_explanation_report.fit(input_id,
                                     explanation_report_start_time,
                                     explanation_report_execution_time,
                                     raw_text,
                                     cleaned_text,
                                     preprocessed_text,
                                     tokens,
                                     original_probabilities.tolist(),
                                     original_label,
                                     expected_label,
                                     flag_pos, flag_sen, flag_mlwe, flag_combinations,
                                     local_explanations)

        # Save the LocalExplanationReport as json on disk in the LocalExplanationReport
        local_explanation_report.save_local_explanation_report(local_explanations_folder, input_name)

        return

    def __prepare_output_folder(self, output_folder, config, timestamp):
        """ Creates the tree output where save each explanation report.  """
        # If the output_folder is not specified as parameter, then it is taken from the configuration file
        if output_folder is None:
            output_folder = config["output_folder"]

        if not os.path.isdir(output_folder):
            print("INFO: Output folder: {} not exists, then it is created.".format(output_folder))
            os.mkdir(output_folder)

        model_output_folder = os.path.join(output_folder, self.model_name)
        if not os.path.isdir(model_output_folder):
            print("INFO: Output Model folder: {} not exists, then it is created inside {}.".format(model_output_folder, output_folder))
            os.mkdir(output_folder)

        model_global_explanations_folder = os.path.join(model_output_folder, "global_explanations_experiments")
        model_local_explanations_folder = os.path.join(model_output_folder, "local_explanations_experiments")

        if not os.path.isdir(model_global_explanations_folder):
            print("INFO: Output global model explanations folder created: {} .".format(model_global_explanations_folder))
            os.mkdir(model_global_explanations_folder)

        if not os.path.isdir(model_local_explanations_folder):
            print("INFO: Output local model explanations folder created: {} .".format(model_local_explanations_folder))
            os.mkdir(model_local_explanations_folder)

        base_experiment_folder = os.path.join(model_local_explanations_folder, timestamp)
        local_explanations_folder = os.path.join(base_experiment_folder, "local_explanations")
        try:
            os.mkdir(base_experiment_folder)
            print("INFO: Output base experiment created: {} .".format(base_experiment_folder))
        except OSError:
            print("Creation of the directory %s failed" % base_experiment_folder)
        try:
            os.mkdir(local_explanations_folder)
            print("INFO: Output local explanations folder created inside the base experiment folder: {} .".format(local_explanations_folder))
        except OSError:
            print("Creation of the directory %s failed" % local_explanations_folder)
        return base_experiment_folder, local_explanations_folder

    @staticmethod
    def __check_fit_parameters(input_texts, classes_of_interest, expected_labels, input_names):
        if not isinstance(input_texts, list):
            raise ValueError("The parameter 'raw_text_list' must be of type: list")
        if any(not isinstance(text, str) for text in input_texts):
            raise ValueError("Each element of the parameter 'raw_text_list' must be of type: string")
        if not isinstance(classes_of_interest, list):
            raise ValueError("The parameter 'class_of_interest_list' must be of type: list")
        if any(not isinstance(c, int) for c in classes_of_interest):
            raise ValueError("Each element of the parameter 'class_of_interest_list' must be of type: int")
        if input_names is not None:
            if not isinstance(input_names, list):
                raise ValueError(" The optional parameter 'input_names' must be of type: list (if specified)")
            if not len(input_texts) == len(classes_of_interest) == len(input_names):
                raise ValueError(
                    "The parameters 'input_names', 'classes_of_interest', 'input_names'  must have all the same length")
        else:
            if not len(input_texts) == len(classes_of_interest):
                raise ValueError("The parameters 'input_names', 'classes_of_interest'  must have all the same length")

        return

    @staticmethod
    def __check_transform_parameters(flag_pos, flag_sen, flag_mlwe, flag_combinations):
        """ Checks parameters of transform method.  """
        if not isinstance(flag_pos, bool):
            raise ValueError("The optional parameter 'flag_pos' must be of type: boolean")
        if not isinstance(flag_sen, bool):
            raise ValueError("The optional parameter 'flag_sen' must be of type: boolean")
        if not isinstance(flag_mlwe, bool):
            raise ValueError("The optional parameter 'flag_mlwe' must be of type: boolean")
        if not isinstance(flag_combinations, bool):
            raise ValueError("The optional parameter 'flag_combinations' must be of type: boolean")
        return


class LocalExplanationReport:
    """ LocalExplanationReport class.  """
    def __init__(self):
        self.report_id = None
        self.start_time = None,
        self.execution_time = None
        self.raw_text = None
        self.cleaned_text = None
        self.preprocessed_text = None
        self.positions_tokens = None
        self.original_probabilities = None
        self.original_label = None
        self.expected_label = None
        self.flag_pos = None
        self.flag_sen = None
        self.flag_mlwe = None
        self.flag_combinations = None
        self.local_explanations = []
        return

    def fit(self, report_id, start_time, execution_time, raw_text, cleaned_text, preprocessed_text, positions_tokens,
            original_probabilities, original_label, expected_label, flag_pos, flag_sen, flag_mlwe, flag_combinations, local_explanations):
        self.report_id = report_id
        self.start_time = start_time,
        self.execution_time = execution_time
        self.raw_text = raw_text
        self.cleaned_text = cleaned_text
        self.preprocessed_text = preprocessed_text
        self.positions_tokens = positions_tokens
        self.original_probabilities = original_probabilities
        self.original_label = original_label
        self.expected_label = expected_label
        self.flag_pos = flag_pos
        self.flag_sen = flag_sen
        self.flag_mlwe = flag_mlwe
        self.flag_combinations = flag_combinations
        self.local_explanations = local_explanations  # List of LocalExplanation
        return

    def add_local_explanations(self, local_explanations):
        self.local_explanations = self.local_explanations + local_explanations
        return

    def save_local_explanation_report(self, output_path, input_name=None):
        """ Save the local explanation report to disk. """

        if input_name is not None:
            report_name = "local_explanation_report_{}_{}.json".format(str(input_name), self.report_id)
        else:
            report_name = "local_explanation_report_{}.json".format(self.report_id)

        # Convert the local explanation report class into a dictionary
        explanation_report_dict = self.local_explanation_report_to_dict()

        with open(os.path.join(output_path, report_name), "w") as fp:
            json.dump(explanation_report_dict, fp)
        return

    def local_explanation_report_metadata_to_dict(self):
        metadata = {"report_id": self.report_id,
                    "start_time": self.start_time,
                    "execution_time": self.execution_time,
                    "flag_pos": self.flag_pos,
                    "flag_sen": self.flag_sen,
                    "flag_mlwe": self.flag_mlwe,
                    "flag_combinations": self.flag_combinations,
                    }
        return metadata

    def local_explanation_report_input_info_to_dict(self):
        input_info = {"raw_text": self.raw_text,
                      "cleaned_text": self.cleaned_text,
                      "preprocessed_text": self.preprocessed_text,
                      "positions_tokens": self.positions_tokens,
                      "original_probabilities": self.original_probabilities,
                      "original_label": self.original_label,
                      "expected_label": self.expected_label
                      }
        return input_info

    def local_explanation_report_to_dict(self):
        """ Converts a single local explanation report to dictionary. """
        metadata = self.local_explanation_report_metadata_to_dict()

        input_info = self.local_explanation_report_input_info_to_dict()

        local_explanations_dict = [local_explanation.local_explanation_to_dict() for local_explanation in
                                   self.local_explanations]

        local_explanation_report_dict = {"metadata": metadata, "input_info": input_info,
                                         "local_explanations": local_explanations_dict}

        return local_explanation_report_dict

    def fit_local_explanation_report_from_json_file(self, explanation_report_path):
        with open(explanation_report_path) as explanation_report_json:
            explanation_report_dict = json.load(explanation_report_json)
            self.fit_local_explanation_report_from_dict(explanation_report_dict)
        return

    def fit_local_explanation_report_from_dict(self, local_explanation_report_dict):

        self.report_id = local_explanation_report_dict["metadata"]["report_id"]
        self.start_time = local_explanation_report_dict["metadata"]["start_time"]
        self.execution_time = local_explanation_report_dict["metadata"]["execution_time"]
        self.flag_pos = local_explanation_report_dict["metadata"]["flag_pos"]
        self.flag_sen = local_explanation_report_dict["metadata"]["flag_sen"]
        self.flag_mlwe = local_explanation_report_dict["metadata"]["flag_mlwe"]
        self.flag_combinations = local_explanation_report_dict["metadata"]["flag_combinations"]

        self.raw_text = local_explanation_report_dict["input_info"]["raw_text"]
        self.cleaned_text = local_explanation_report_dict["input_info"]["cleaned_text"]
        self.preprocessed_text = local_explanation_report_dict["input_info"]["preprocessed_text"]
        self.positions_tokens = local_explanation_report_dict["input_info"]["positions_tokens"]
        self.original_probabilities = local_explanation_report_dict["input_info"]["original_probabilities"]
        self.original_label = local_explanation_report_dict["input_info"]["original_label"]
        self.expected_label = local_explanation_report_dict["input_info"]["expected_label"]

        local_explanation_list = local_explanation_report_dict["local_explanations"]
        self.local_explanations = [self.create_local_explanation_from_dict(local_explanation_dict) for local_explanation_dict in local_explanation_list]
        print("\n")
        return

    def get_filtered_local_explanations(self, feature_type_list=["MLWE", "POS", "SEN"], combination_list=[1, 2]):
        return self.filter_local_explanations(self.local_explanations, feature_type_list, combination_list)

    @staticmethod
    def create_local_explanation_from_dict(local_explanation_dict):
        local_explanation = le.LocalExplanation()
        local_explanation.fit_from_dict(local_explanation_dict)
        return local_explanation

    def get_most_influential_feature(self, class_of_interest=-1, comb_list=[1, 2]):
        if comb_list != [1, 2]:
            local_explanations = self.filter_local_explanations_by_feature_combination(self.local_explanations, comb_list)
        else:
            local_explanations = self.local_explanations
        most_influential_feature = self.filter_local_explanations_by_max_nPIR(local_explanations, class_of_interest)
        return most_influential_feature

    def get_mlwe_most_influential_feature(self, class_of_interest=-1, comb_list=[1, 2]):
        mlwe_local_explanations = self.filter_local_explanations_by_feature_type(self.local_explanations, "MLWE")
        if comb_list != [1, 2]:
            mlwe_local_explanations = self.filter_local_explanations_by_feature_combination(mlwe_local_explanations, comb_list)
        most_influential_mlwe_feature = self.filter_local_explanations_by_max_nPIR(mlwe_local_explanations, class_of_interest)
        return most_influential_mlwe_feature

    def get_sen_most_influential_feature(self, class_of_interest=-1, comb_list=[1, 2]):
        sen_local_explanations = self.filter_local_explanations_by_feature_type(self.local_explanations, "SEN")
        if comb_list != [1, 2]:
            sen_local_explanations = self.filter_local_explanations_by_feature_combination(sen_local_explanations, comb_list)
        most_influential_sen_feature = self.filter_local_explanations_by_max_nPIR(sen_local_explanations, class_of_interest)
        return most_influential_sen_feature

    def get_pos_most_influential_feature(self, class_of_interest=-1, comb_list=[1, 2]):
        pos_local_explanations = self.filter_local_explanations_by_feature_type(self.local_explanations, "POS")
        if comb_list != [1, 2]:
            pos_local_explanations = self.filter_local_explanations_by_feature_combination(pos_local_explanations, comb_list)
        most_influential_pos_feature = self.filter_local_explanations_by_max_nPIR(pos_local_explanations, class_of_interest)
        return most_influential_pos_feature

    @staticmethod
    def filter_local_explanations_by_feature_type(local_explanations, feature_type):
        """ Gets all the """
        return [local_explanation for local_explanation in local_explanations if local_explanation.perturbation.feature.feature_type == feature_type]

    @staticmethod
    def filter_local_explanations_by_feature_combination(local_explanations, combination_list=[1, 2]):
        """ Filters the local_explanation by their feature.combination value. """
        return [local_explanation for local_explanation in local_explanations if local_explanation.perturbation.feature.combination in combination_list]

    @staticmethod
    def filter_local_explanations_by_nPIR_range(local_explanations, nPIR_range=[-1, +1], class_of_interest=-1):
        """ Gets all the """
        if class_of_interest == -1:
            filtered_local_explanation = [l_e for l_e in local_explanations if l_e.nPIR_original_top_class >= nPIR_range[0] and l_e.nPIR_original_top_class >= nPIR_range[1]]
        else:
            coi = class_of_interest
            filtered_local_explanation = [l_e for l_e in local_explanations if l_e.nPIRs[coi] >= nPIR_range[0] and l_e.nPIRs[coi] >= nPIR_range[1]]
        return filtered_local_explanation

    @staticmethod
    def filter_local_explanations_by_max_nPIR(local_explanations, class_of_interest=-1):
        """ Gets all the """
        if class_of_interest == -1:
            most_influential_feature = max(local_explanations, key=lambda local_explanation: local_explanation.numerical_explanation.nPIR_original_top_class)
        else:
            most_influential_feature = max(local_explanations, key=lambda local_explanation: local_explanation.numerical_explanation.nPIRs.class_of_interest)
        return most_influential_feature

    @staticmethod
    def filter_local_explanations(local_explanations, feature_type_list=["MLWE", "POS", "SEN"], combination_list=[1, 2]):
        return [l_e for l_e in local_explanations if l_e.perturbation.feature.feature_type in feature_type_list and l_e.perturbation.feature.combination in combination_list]


class GlobalExplainer:
    def __init__(self, label_list, label_names=None):
        self.label_list = label_list
        self.label_names = label_names
        self.local_explanation_reports = []

        return

    def fit_from_folder(self, reports_folder_path):
        for file in os.listdir(reports_folder_path):
            if file.endswith(".json"):
                explanation_report_json_path = os.path.join(reports_folder_path, file)

                # instantiate the explanation report
                local_explanation_report = LocalExplanationReport()
                local_explanation_report.fit_local_explanation_report_from_json_file(explanation_report_json_path)

                self.local_explanation_reports.append(local_explanation_report)
        return

    def transform(self, feature_type=any(["MLWE", "POS", "SEN", "ALL"]), skipped_tokens=[]):
        GAI_token_matrix = {}
        GAI_lemma_matrix = {}
        GRI_token_matrix = {}
        GRI_lemma_matrix = {}
        token_occurency_matrix = {}
        lemma_occurency_matrix = {}
        token_indices_input_reports_info = {}  # {"token":List[int] (indices)}
        lemma_indices_input_reports_info = {}
        input_reports_info = []  # [ { "metadata", "input_info", "most_informative_local_explanation"} ]

        number_of_reports = len(self.local_explanation_reports)

        lemmatizer = WordNetLemmatizer()
        report_index = 0
        for local_explanation_report in self.local_explanation_reports:

            metadata = local_explanation_report.local_explanation_report_metadata_to_dict()
            input_info = local_explanation_report.local_explanation_report_input_info_to_dict()

            if feature_type == "MLWE":
                most_influential_feature = local_explanation_report.get_mlwe_most_influential_feature()
            elif feature_type == "POS":
                most_influential_feature = local_explanation_report.get_pos_most_influential_feature()
            elif feature_type == "SEN":
                most_influential_feature = local_explanation_report.get_sen_most_influential_feature()
            else:
                most_influential_feature = local_explanation_report.get_most_influential_feature()

            original_label = local_explanation_report.original_label

            nPIR_most_influential_feature = most_influential_feature.numerical_explanation.nPIRs[original_label]
            tokens_most_influential_feature = list(most_influential_feature.perturbation.feature.positions_tokens.values())

            input_report_info = {"metadata": metadata, "input_info": input_info, "most_informative_local_explanation": most_influential_feature}
            input_reports_info.append(input_report_info)

            for token in tokens_most_influential_feature:
                if token not in skipped_tokens:
                    if token not in GAI_token_matrix.keys():
                        GAI_token_matrix[token] = [0]*len(self.label_list)
                        token_occurency_matrix[token] = [0]*len(self.label_list)
                    GAI_token_matrix[token][original_label] = GAI_token_matrix[token][original_label] + nPIR_most_influential_feature
                    token_occurency_matrix[token][original_label] = token_occurency_matrix[token][original_label] + 1

                    if token not in token_indices_input_reports_info.keys():
                        token_indices_input_reports_info[token] = []
                    token_indices_input_reports_info[token].append(report_index)

                    lemma = lemmatizer.lemmatize(token)
                    if lemma not in GAI_lemma_matrix.keys():
                        GAI_lemma_matrix[lemma] = [0]*len(self.label_list)
                        lemma_occurency_matrix[lemma] = [0]*len(self.label_list)
                    GAI_lemma_matrix[lemma][original_label] = GAI_lemma_matrix[lemma][original_label] + nPIR_most_influential_feature
                    lemma_occurency_matrix[lemma][original_label] = lemma_occurency_matrix[lemma][original_label] + 1

                    if lemma not in lemma_indices_input_reports_info.keys():
                        lemma_indices_input_reports_info[lemma] = []
                    lemma_indices_input_reports_info[lemma].append(report_index)

            report_index += 1

        for token, GAIs in GAI_token_matrix.items():
            GRI_token_matrix[token] = [max(self.compute_GRI(label, GAIs), 0) for label in range(len(GAIs))]

        for lemma, GAIs in GAI_lemma_matrix.items():
            GRI_lemma_matrix[lemma] = [max(self.compute_GRI(label, GAIs), 0) for label in range(len(GAIs))]

        global_explanation_report = GlobalExplanationReport()
        global_explanation_report.fit(self.label_list, self.label_names, number_of_reports, GAI_token_matrix, GRI_token_matrix, token_occurency_matrix,
                                      GAI_lemma_matrix, GRI_lemma_matrix, lemma_occurency_matrix, token_indices_input_reports_info, lemma_indices_input_reports_info,
                                      input_reports_info)
        return global_explanation_report

    @staticmethod
    def compute_GRI(label_id, GAIs):
        gri = 0
        for i in range(len(GAIs)):
            if i != label_id:
                gri = gri - GAIs[i]
            else:
                gri = gri + GAIs[i]
        return gri

    def __check_init_parameters(self):
        return


class GlobalExplanationReport:
    def __init__(self):
        self.label_list = None
        self.label_names = None
        self.number_of_reports = None
        self.GAI_token_matrix = None
        self.GAI_lemma_matrix = None
        self.GRI_token_matrix = None
        self.GRI_lemma_matrix = None
        self.token_occurency_matrix = None
        self.lemma_occurency_matrix = None
        self.token_indices_input_reports_info = None
        self.lemma_indices_input_reports_info = None
        self.input_reports_info = None

        return

    def fit(self, label_list, label_names, number_of_reports, GAI_token_matrix, GRI_token_matrix, token_occurency_matrix, GAI_lemma_matrix,
            GRI_lemma_matrix, lemma_occurency_matrix, token_indices_input_reports_info, lemma_indices_input_reports_info, input_reports_info):
        self.label_list = label_list
        self.label_names = label_names
        self.number_of_reports = number_of_reports
        self.GAI_token_matrix = GAI_token_matrix
        self.GAI_lemma_matrix = GAI_lemma_matrix
        self.GRI_token_matrix = GRI_token_matrix
        self.GRI_lemma_matrix = GRI_lemma_matrix
        self.token_occurency_matrix = token_occurency_matrix
        self.lemma_occurency_matrix = lemma_occurency_matrix
        self.token_indices_input_reports_info = token_indices_input_reports_info
        self.lemma_indices_input_reports_info = lemma_indices_input_reports_info
        self.input_reports_info = input_reports_info

        return

    def get_most_informative_local_explanations_by_token(self, token, label_list=-1):
        if token not in self.token_indices_input_reports_info.keys():
            print("INFO: Token {} not appear in most informative local explanations".format(token))
            return None
        if label_list == -1:
            label_list = self.label_list
        return [self.input_reports_info[i] for i in self.token_indices_input_reports_info[token]
                if self.input_reports_info[i]["most_informative_local_explanation"].original_top_class in label_list]

    def global_explanation_report_to_dict(self):

        input_reports_info_dict = [{"metadata": report_dict["metadata"],
                                    "input_info": report_dict["input_info"],
                                    "most_informative_local_explanation": report_dict["most_informative_local_explanation"].local_explanation_to_dict(),
                                    } for report_dict in self.input_reports_info]

        global_explanation_dictionary = {
            "label_list": self.label_list,
            "label_names": self.label_names,
            "number_of_reports": self.number_of_reports,
            "GAI_token_matrix": self.GAI_token_matrix,
            "GAI_lemma_matrix": self.GAI_lemma_matrix,
            "GRI_token_matrix": self.GRI_token_matrix,
            "GRI_lemma_matrix": self.GRI_lemma_matrix,
            "token_occurency_matrix": self.token_occurency_matrix,
            "lemma_occurency_matrix": self.lemma_occurency_matrix,
            "token_indices_input_reports_info": self.token_indices_input_reports_info,
            "lemma_indices_input_reports_info": self.lemma_indices_input_reports_info,
            "input_reports_info": input_reports_info_dict
        }
        return global_explanation_dictionary

    def dict_to_global_explanation_report(self, global_explanation_dictionary):
        self.label_list = global_explanation_dictionary["label_list"]
        self.label_names = global_explanation_dictionary["label_names"]
        self.number_of_reports = global_explanation_dictionary["number_of_reports"]
        self.GAI_token_matrix = global_explanation_dictionary["GAI_token_matrix"]
        self.GAI_lemma_matrix = global_explanation_dictionary["GAI_lemma_matrix"]
        self.GRI_token_matrix = global_explanation_dictionary["GRI_token_matrix"]
        self.GRI_lemma_matrix = global_explanation_dictionary["GRI_lemma_matrix"]
        self.token_occurency_matrix = global_explanation_dictionary["token_occurency_matrix"]
        self.lemma_occurency_matrix = global_explanation_dictionary["lemma_occurency_matrix"]
        self.token_indices_input_reports_info = global_explanation_dictionary["token_indices_input_reports_info"]
        self.lemma_indices_input_reports_info = global_explanation_dictionary["lemma_indices_input_reports_info"]
        self.input_reports_info = [{"metadata": report_dict["metadata"],
                                    "input_info": report_dict["input_info"],
                                    "most_informative_local_explanation": le.LocalExplanation.fit_from_dict(report_dict["most_informative_local_explanation"])}
                                   for report_dict in global_explanation_dictionary["input_reports_info"]]
        return

    def save_global_explanation_report(self, output_path, input_name=None):
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')

        if input_name is not None:
            report_name = "global_explanation_report_{}_{}.json".format(str(input_name), timestamp)
        else:
            report_name = "global_explanation_report_{}.json".format(timestamp)

        # Convert the global explanation report class into a dictionary
        global_explanation_report_dict = self.global_explanation_report_to_dict()

        with open(os.path.join(output_path, report_name), "w") as fp:
            json.dump(global_explanation_report_dict, fp)
        return


