import feature_extraction as fe
import perturbation as pe
import local_explanation as le
from typing import List

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
        self.input_names = None

    def fit(self, input_texts:  List[str], classes_of_interest: List[int], input_names: List[any] = None):
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
            input_names (List[str])[Optional]: Fits the explainer with a list of names (one for each input). If passed, this information will
                                               be used to compose the local explanation report name.
        Raises:
            ValueError: if `input_texts` is not a list of strings
            ValueError: if `classes_of_interest` is not a list of integers
            ValueError: if `input_names` is not a list of integers (if passed as parameter)
            ValueError: if `len(input_texts)` and `len(classes_of_interest)` are not equal
        """
        self.__check_fit_parameters(input_texts, classes_of_interest, input_names)  # Check parameters of fit method
        if input_names is None:
            self.input_names = [None]*len(input_texts)
        else:
            self.input_names = input_names
        self.raw_texts = input_texts  # List[str] containing the original raw input texts
        self.classes_of_interest = classes_of_interest  # List[int] containing the class_of_interest for the explanation of each input
        self.cleaned_texts = [self.model_wrapper.clean_function(text) for text in input_texts]  # List[str] Clean each input with the clean_function specified in the model_wrapper
        self.sequences = self.model_wrapper.texts_to_sequences(input_texts)  # List[List[int]] list of sequences ids (one sequence of ids for each input text)
        self.tokens_list = self.model_wrapper.texts_to_tokens(input_texts)  # List[List[str]] list of sequences tokens (one sequence of tokens for each input text)
        self.preprocessed_texts = self.model_wrapper.sequences_to_texts(self.sequences)  # List[str] list of texts by joining each sequence of tokens
        return

    def transform(self, flag_pos: bool = True, flag_sen: bool = True, flag_mlwe: bool = True, flag_combinations: bool = True, output_folder: str = None):
        """ Create a local explanation report for each input text passed in the fit method.

        For each text passed in the fit method, it perform the local explanation for the relative class of interest and
        save one explanation report of each input text separately. The types of feature extraction methods performed are
        based on the boolean parameters passed.

        Args:
            flag_pos (bool): True if want perform Part-of-Speech Feature Extraction, False otherwise
            flag_sen (bool): True if want perform Sentence Feature Extraction, False otherwise
            flag_mlwe (bool): True if want perform Multi-Layer Word Embedding Feature Extraction, False otherwise
            flag_combinations (bool): True if want perform (pairwise) combinations of features, False otherwise
            output_folder (str)[Optional]: folder where save the outputs, the default folder is specified in the config.yaml file
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
        for raw_text, cleaned_text, preprocessed_text, tokens, class_of_interest, embedding_tensor, input_name in \
                zip(self.raw_texts, self.cleaned_texts, self.preprocessed_texts, self.tokens_list, self.classes_of_interest, embedding_tensors, self.input_names):
            self.__perform_local_explanation_single_input_text(input_id,  # Will fill the report id number
                                                               raw_text,
                                                               cleaned_text,
                                                               preprocessed_text,
                                                               tokens,
                                                               class_of_interest,
                                                               embedding_tensor,
                                                               flag_pos, flag_sen, flag_mlwe, flag_combinations,
                                                               local_explanations_folder,
                                                               input_name)
            input_id += 1

        return

    def fit_transform(self, input_texts, classes_of_interest, flag_pos, flag_sen, flag_mlwe, flag_combinations):
        """ Fits the explainer with input texts and perform transform methods to create the local explanation reports.

        """
        self.fit(input_texts, classes_of_interest)

        self.transform(flag_pos, flag_sen, flag_mlwe, flag_combinations)
        return

    def __perform_local_explanation_single_input_text(self, input_id, raw_text, cleaned_text, preprocessed_text, tokens,
                                                      class_of_interest, embedding_tensor,
                                                      flag_pos, flag_sen, flag_mlwe, flag_combinations, local_explanations_folder, input_name):
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

        # Perturb the features
        perturbation_manager.execute_perturbation_phase()

        # Get the list of all the perturbations produced -> List[Perturbation]
        perturbations = perturbation_manager.get_all_perturbations()

        # Instantiate the LocalExplanationManager
        local_explanation_manager = le.LocalExplanationManager(preprocessed_text,  # Preprocessed version of the current input text
                                                               self.model_wrapper,
                                                               class_of_interest,  # Class of interest for the current input
                                                               perturbations)  # List of Perturbations to which produce the local explanation

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
                                     original_probabilities,
                                     original_label,
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
    def __check_fit_parameters(input_texts, classes_of_interest, input_names):
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
        self.flag_pos = None
        self.flag_sen = None
        self.flag_mlwe = None
        self.flag_combinations = None
        self.local_explanations = []
        return

    def fit(self, report_id, start_time, execution_time, raw_text, cleaned_text, preprocessed_text, positions_tokens,
            original_probabilities, original_label, flag_pos, flag_sen, flag_mlwe, flag_combinations, local_explanations):
        self.report_id = report_id
        self.start_time = start_time,
        self.execution_time = execution_time
        self.raw_text = raw_text
        self.cleaned_text = cleaned_text
        self.preprocessed_text = preprocessed_text
        self.positions_tokens = positions_tokens
        self.original_probabilities = original_probabilities
        self.original_label = original_label
        self.flag_pos = flag_pos
        self.flag_sen = flag_sen
        self.flag_mlwe = flag_mlwe
        self.flag_combinations = flag_combinations
        self.local_explanations = local_explanations # List of LocalExplanation
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

    def local_explanation_report_to_dict(self):
        """ Converts a single local explanation report to dictionary. """
        metadata = {"report_id": self.report_id,
                    "start_time": self.start_time,
                    "execution_time": self.execution_time,
                    "flag_pos": self.flag_pos,
                    "flag_sen": self.flag_sen,
                    "flag_mlwe": self.flag_mlwe,
                    "flag_combinations": self.flag_combinations,
                    }
        input_info = {"raw_text": self.raw_text,
                      "cleaned_text": self.cleaned_text,
                      "preprocessed_text": self.preprocessed_text,
                      "positions_tokens": self.positions_tokens,
                      "original_probabilities": self.original_probabilities.tolist(),
                      "original_label": self.original_label}

        local_explanations_dict = [self.local_explanation_to_dict(local_explanation) for local_explanation in
                                   self.local_explanations]

        local_explanation_report_dict = {"metadata": metadata, "input_info": input_info,
                                         "local_explanations": local_explanations_dict}

        return local_explanation_report_dict

    @staticmethod
    def local_explanation_to_dict(local_explanation):
        """ Converts a single local explanation into dictionary. """
        perturbation = local_explanation.perturbation
        feature = perturbation.feature
        local_explanation_dict = {
            "local_explanation_id": local_explanation.local_explanation_id,
            "feature_id": feature.feature_id,
            "feature_type": feature.feature_type,
            "feature_description": feature.description,
            "positions_tokens": feature.positions_tokens,
            "combination": feature.combination,
            "perturbation_id": perturbation.perturbation_id,
            "perturbation_type": perturbation.perturbation_type,
            "perturbed_text": perturbation.perturbed_text,
            "original_probabilities": local_explanation.original_probabilities.tolist(),
            "perturbed_probabilities": local_explanation.perturbed_probabilities.tolist(),
            "original_top_class": local_explanation.original_top_class,
            "perturbed_top_class": local_explanation.perturbed_top_class,
            "class_of_interest": local_explanation.class_of_interest,
            "nPIR_original_top_class": local_explanation.numerical_explanation.nPIR_original_top_class,
            "nPIRP_original_top_class": local_explanation.numerical_explanation.nPIRP_original_top_class,
            "nPIR_class_of_interest": local_explanation.numerical_explanation.nPIR_class_of_interest,
            "nPIRP_class_of_interest": local_explanation.numerical_explanation.nPIRP_class_of_interest,
            "nPIRs": local_explanation.numerical_explanation.nPIRs,
            "nPIRPs": local_explanation.numerical_explanation.nPIRPs,
            "k": local_explanation.perturbation.feature.k
        }
        return local_explanation_dict

    def fit_local_explanation_report_from_json_file(self, explanation_report_path):
        with open(explanation_report_path) as explanation_report_json:
            explanation_report_dict = json.load(explanation_report_json)
            self.dict_to_local_explanation_report(explanation_report_dict)
        return

    def dict_to_local_explanation_report(self, local_explanation_report_dict):

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

        local_explanation_list = local_explanation_report_dict["local_explanations"]
        self.local_explanations = [self.dict_to_local_explanation(local_explanation_dict) for local_explanation_dict in local_explanation_list]

        return

    @staticmethod
    def dict_to_local_explanation(local_explanation_dict):
        feature = fe.Feature(local_explanation_dict["feature_id"],
                             local_explanation_dict["feature_type"],
                             local_explanation_dict["feature_description"],
                             local_explanation_dict["positions_tokens"],
                             local_explanation_dict["combination"])

        perturbation = pe.Perturbation(local_explanation_dict["perturbation_id"],
                                       local_explanation_dict["perturbation_type"],
                                       local_explanation_dict["perturbed_text"],
                                       feature)

        numerical_explanation = le.NumericalExplanation(local_explanation_dict["nPIR_original_top_class"],
                                                        local_explanation_dict["nPIRP_original_top_class"],
                                                        local_explanation_dict["nPIR_class_of_interest"],
                                                        local_explanation_dict["nPIRP_class_of_interest"],
                                                        local_explanation_dict["nPIRs"],
                                                        local_explanation_dict["nPIRPs"])

        local_explanation = le.LocalExplanation(local_explanation_dict["local_explanation_id"],
                                                perturbation,
                                                local_explanation_dict["original_probabilities"],
                                                local_explanation_dict["perturbed_probabilities"],
                                                local_explanation_dict["original_top_class"],
                                                local_explanation_dict["perturbed_top_class"],
                                                local_explanation_dict["class_of_interest"],
                                                numerical_explanation)
        return local_explanation


class GlobalExplainer:
    def __init__(self, label_list, label_names=None):
        self.label_list = label_list
        self.label_names = label_names
        self.local_explanation_reports = []
        pass

    def fit_from_folder(self, reports_folder_path):
        for file in os.listdir(reports_folder_path):
            if file.endswith(".json"):
                explanation_report_json_path = os.path.join(reports_folder_path, file)

                # instantiate the explanation report
                local_explanation_report = LocalExplanationReport()
                local_explanation_report.fit_local_explanation_report_from_json_file(explanation_report_json_path)

                self.local_explanation_reports.append(local_explanation_report)
        return

    def transform(self, flag_pos=False, flag_sen=False, flag_mlwe=False, flag_combinations=True, discarded_tokens=[]):
        for report in self.local_explanation_reports:
            print(report.report_id)
            print(report.raw_text)
            print("\n")
        return

    def __check_init_parameters(self):
        return


class GlobalExplanationReport:
    def __init__(self):
        pass





