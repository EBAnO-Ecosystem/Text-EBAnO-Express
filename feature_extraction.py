import perturbation as pe
import local_explanation as le
from abc import ABC, abstractmethod
from typing import List
from itertools import combinations
from sklearn.cluster import KMeans

import numpy as np
import nltk
import yaml
import math


class FeaturesExtractionManager:
    """ Features Extraction Manager Class.
    The FeaturesExtractionPhase class manages the workflow of the feature extraction phase in the explanation process.

     Attributes:
        raw_text (str): String containing the raw input text (not preprocessed)
        cleaned_text (str): String containing the clean version of the input text (calling the method model_wrapper.clean_text(raw_text))
        preprocessed_text (str): String created by tokenizing and joining tokens (can contains <OOV> tokens)
        flag_pos (Optional[bool]): True if want extract Part-Of-Speech features, False otherwise
        flag_sen (Optional[bool]): True if want extract Sentence features, False otherwise
        flag_mlwe (Optional[bool]): True if want extract MultiLayer Word Embedding features, False otherwise
        flag_combinations (Optional[bool]): True if want extract pairwise Combinations of features, False otherwise
        pos_features_extraction_method (:obj:PartsOfSpeechFeaturesExtraction): Object of type PartsOfSpeechFeaturesExtraction
        sen_features_extraction_method (:obj:SentencesFeaturesExtraction): Object of type SentencesFeaturesExtraction
        mlwe_features_extraction_method (:obj:MultiLayerWordEmbeddingFeaturesExtraction): Object of type MultiLayerWordEmbeddingFeaturesExtraction
        pos_features (list[Feature]): List of features extracted with the Part-Of-Speech feature extraction method
        sen_features (list[Feature]): List of features extracted with the Sentence feature extraction method
        mlwe_features (list[Feature]): List of features extracted with the MultiLayer Word Embedding feature extraction method
    """
    def __init__(self, raw_text, cleaned_text, preprocessed_text, tokens, class_of_interest, embedding_tensor, model_wrapper,
                 flag_pos=True, flag_sen=True, flag_mlwe=True, flag_combinations=True):
        """ FeaturesExtractionPhase Initializer.
        Args:
            raw_text (str):  String containing the raw input text (not preprocessed)
            cleaned_text (str):
            preprocessed_text (str):
            model_wrapper (:obj:ModelWrapperInterface)  Instance of a real class implementing the ModelWrapperInterface
            flag_pos (Optional[bool]): True if want extract Part-Of-Speech features, False otherwise
            flag_sen (Optional[bool]): True if want extract Sentence features, False otherwise
            flag_mlwe (Optional[bool]): True if want extract MultiLayer Word Embedding features, False otherwise
            flag_combinations (Optional[bool]):
        """
        self.raw_text = raw_text
        self.cleaned_text = cleaned_text
        self.preprocessed_text = preprocessed_text
        self.tokens = tokens
        self.class_of_interest = class_of_interest
        self.embedding_tensor = embedding_tensor
        self.model_wrapper = model_wrapper
        self.flag_pos = flag_pos
        self.flag_sen = flag_sen
        self.flag_mlwe = flag_mlwe
        self.flag_combinations = flag_combinations
        self.pos_features_extraction_method = None
        self.sen_features_extraction_method = None
        self.mlwe_features_extraction_method = None
        self.pos_features = []  # List[:obj:Feature] extracted with the PartsOfSpeechFeaturesExtraction
        self.sen_features = []  # List[:obj:Feature] extracted with the SentencesFeaturesExtraction
        self.mlwe_features = []  # List[:obj:Feature] extracted with the MultiLayerWordEmbeddingFeaturesExtraction

    def execute_feature_extraction_phase(self):
        """ Execute the Feature Extraction Phase. """
        if self.flag_pos:
            # If flag_pos is True, then instantiate the PartsOfSpeechFeaturesExtraction and extract POS features
            self.pos_features_extraction_method = PartsOfSpeechFeaturesExtraction(self.raw_text, self.cleaned_text, self.preprocessed_text,
                                                                                  self.tokens, self.class_of_interest, self.model_wrapper, self.flag_combinations)
            self.pos_features = self.pos_features_extraction_method.extract_features()

        if self.flag_sen:
            # If flag_sen is True, then instantiate the SentencesFeaturesExtraction and extract SEN features
            self.sen_features_extraction_method = SentencesFeaturesExtraction(self.raw_text, self.cleaned_text, self.preprocessed_text,
                                                                              self.tokens, self.class_of_interest, self.model_wrapper, self.flag_combinations)
            self.sen_features = self.sen_features_extraction_method.extract_features()

        if self.flag_mlwe:
            # If flag_mlwe is True, then instantiate the MultiLayerWordEmbeddingFeaturesExtraction and extract MLWE features
            self.mlwe_features_extraction_method = MultiLayerWordEmbeddingFeaturesExtraction(self.raw_text, self.cleaned_text, self.preprocessed_text,
                                                                                             self.tokens, self.class_of_interest, self.embedding_tensor,
                                                                                             self.model_wrapper, self.flag_combinations)
            self.mlwe_features = self.mlwe_features_extraction_method.extract_features()
        return

    def get_pos_features(self):
        """ Returns: (list[Feature]) List containing all the extracted Part-Of-Speech features. """
        return self.pos_features

    def get_sen_features(self):
        """ Returns: (list[Feature]) List containing all the extracted Sentence features. """
        return self.sen_features

    def get_mlwe_features(self):
        """ Returns: (list[Feature]) List containing all the extracted MultiLayer Word Embedding features. """
        return self.mlwe_features

    def get_all_features(self):
        """ Returns: (list[Feature]) List containing all the extracted features (with all feature extraction methods). """
        return self.pos_features + self.sen_features + self.mlwe_features


class Feature:
    """ Feature Class: a Feature represents a single feature extracted. """
    def __init__(self, feature_id, feature_type, description, positions_tokens, combination=1, k=None):
        """ Feature Initializer.
        Args:
            feature_id (int): feature identifier
            feature_type (str): string containing the feature method type (POS, SEN or MLWE)
            description (str): string containing the description of the feature (e.g., Adjectives, Nouns, Sentence1, Cluster1)
            positions_tokens (list[]):
            combination (int): number of combinations to create the feature (1 if no combination, 2 for pairwise combination)
            k (int): (only for MLWE) specifies the K value to which has been performed the k-means, None for POS and SEN
        """
        self.feature_id = feature_id
        self.feature_type = feature_type  # POS , SEN , MLWE
        self.description = description
        self.positions_tokens = positions_tokens
        self.combination = combination
        self.k = k

    def print_feature_info(self):
        """ Print information about the Feature. """
        print("Feature ID: ", self.feature_id)
        print("Feature Extraction Method: ", self.feature_type)
        print("Description: ", self.description)
        print("Position-Token Tuples: ", self.positions_tokens)
        print("Combination: ", self.combination)
        return

    def get_feature_id(self):
        """ Returns: (int) feature id. """
        return self.feature_id

    def get_feature_extraction_method(self):
        """ Returns: (str) feature extraction method used. """
        return self.feature_type

    def get_feature_description(self):
        """ Returns: (int) feature description. """
        return self.description

    def get_list_positions_tokens(self):
        return self.positions_tokens


class FeaturesExtractionMethod(ABC):
    """ Abstract Class: Features Extraction Method. """
    def __init__(self, raw_text, cleaned_text, preprocessed_text, tokens, class_of_interest, model_wrapper, flag_combinations):
        """
        Args:
            raw_text
            cleaned_text
            preprocessed_text
            tokens
        """
        self.raw_text = raw_text
        self.cleaned_text = cleaned_text
        self.preprocessed_text = preprocessed_text
        self.tokens = tokens
        self.class_of_interest = class_of_interest
        self.model_wrapper = model_wrapper
        self.flag_combinations = flag_combinations
        self.feature_extraction_type = None

    @abstractmethod
    def extract_features(self) -> List[Feature]:
        """ Abstract Method: Extract features from the Input Text. """
        pass

    def combine_feature(self, features, r, start_id, feature_type):
        """ Create r-wise combination of features.

        features (List[:obj:Feature]): List of features to be combined
        r (int): apply r-wise combination (r=2 for pairwise combinations)
        start_id (int):
        feature_type (str): String containing the feature type `POS`,`SEN`,`MLWE`
        """
        combination_features = []
        feature_id = start_id
        for subset_features in list(combinations(features, r)):

            descriptions = [feature.description for feature in list(subset_features)]
            description = self.create_combination_description(descriptions)

            positions_tokens = {k: v for sublist in list(subset_features) for k, v in sublist.positions_tokens.items()}
            feature = self.fit_combination_feature(feature_id, feature_type, description, positions_tokens, r)
            combination_features.append(feature)
            feature_id += 1
        return combination_features

    @staticmethod
    def fit_combination_feature(feature_id, feature_type, description, positions_tokens, combination):
        feature = Feature(feature_id,
                          feature_type,
                          description,
                          positions_tokens,
                          combination)
        return feature

    @staticmethod
    def create_combination_description(descriptions):
        description = ""
        if len(descriptions) < 2:
            description = descriptions[0]
        if len(descriptions) == 2:
            description = "Combination of {} and {}".format(descriptions[0],descriptions[1])
        if len(descriptions) > 2:
            description = "Combination of {},".format(descriptions[0])
            for i in range(1,len(descriptions)-2):
                description = "{} {},".format(description, descriptions[i])
            description = "{}, {} and {}".format(description, descriptions[len(descriptions)-2], descriptions[len(descriptions)-1])
        return description


class PartsOfSpeechFeaturesExtraction(FeaturesExtractionMethod):
    """ Part-Of-Speech Feature Extraction Class: Implementation of the FeaturesExtractionMethod Abstract Class.

        The Part-Of-Speech feature extraction method extracts one feature for each part-of-speech analyzed (listed in the configuration file `pos_configuration.yaml`).
        For examples: the list of Adjectives will be the first feature, the list of Nouns the second feature, the list of Verbs the third feature and so on.
    """
    def __init__(self, raw_text, cleaned_text, preprocessed_text, tokens, class_of_interest, model_wrapper, flag_combinations):
        # Execute constructor of the FeaturesExtractionMethod (Father Class)
        FeaturesExtractionMethod.__init__(self, raw_text, cleaned_text, preprocessed_text, tokens, class_of_interest, model_wrapper, flag_combinations)
        return

    def extract_features(self):
        self.feature_extraction_type = "POS"

        # Read Part-of-speech configuration file --> it contains the list of pos analyzed by T-EBAnO
        config = self.__read_configuration_file()
        pos_scheduled = self.__parse_pos_configuration(config)

        # Add the part-of-speech tag and the position to each token
        tokens_tags = self.tag_input_text(self.tokens)
        tokens_tags_positions = self.add_position(tokens_tags)

        # Create one feature for each part-of-speech analyzed
        features = []
        feature_id = 0
        # Extract one feature for each part-of-speech
        for pos in pos_scheduled:
            feature = self.fit_feature(feature_id,
                                       tokens_tags_positions,
                                       pos,
                                       1)
            features.append(feature)
            feature_id += 1

        if self.flag_combinations is True:
            combination_features = self.combine_feature(features, 2, len(features), self.feature_extraction_type)
            features = features + combination_features

        return features

    def fit_feature(self, feature_id, tokens_tags_positions, pos, combination):
        positions_tokens = {}
        for token_tag_position in tokens_tags_positions:
            if token_tag_position[1] in pos["tags"]:
                positions_tokens[token_tag_position[2]] = token_tag_position[0]

        feature = Feature(feature_id,
                          self.feature_extraction_type,
                          self.create_description(pos["description"]),
                          positions_tokens,
                          combination)

        return feature

    @staticmethod
    def tag_input_text(tokens):
        return nltk.pos_tag(tokens)

    @staticmethod
    def add_position(tokens_tags):
        return [list(token_tag) + [i] for i, token_tag in zip(range(len(tokens_tags)), tokens_tags)]

    @staticmethod
    def create_description(pos_name):
        return str(pos_name)

    @staticmethod
    def __read_configuration_file():
        """ Read the part-of-speech configuration file containing the list of part-of-speech and relative tags analyzed.

        Returns:
            config (dict): dictionary with pos scheduled and tags for each pos
        """
        with open(r'config_files/pos_configuration.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config

    @staticmethod
    def __parse_pos_configuration(config):
        """ Parse the configuration file into a list of part-of-speech to be analyzed.

        Returns:
            pos_scheduled (list(dict)): list of dict, each dict contains:
                        ['description'] (str) : part-of-speech name
                        ['tags'] (list(str)): list of tags of the current part-of-speech.
        """
        pos_scheduled = []
        for pos_dict in config["scheduling"]:
            pos_name = next(iter(pos_dict))
            current_pos = {"description": pos_name, "tags": []}
            for tag_group in list(pos_dict.values())[0]:
                current_pos["tags"] += config[tag_group]
            pos_scheduled.append(current_pos)
        return pos_scheduled


class SentencesFeaturesExtraction(FeaturesExtractionMethod):
    """ Sentences Feature Extraction Class: Implementation of the FeaturesExtractionMethod Abstract Class.

    The Sentences feature extraction method extracts one feature for each sentence in the input text.
    For examples: the first sentence will be the first feature, the second sentence will be the second feature and so on.
    """
    def __init__(self, raw_text, cleaned_text, preprocessed_text, tokens, class_of_interest, model_wrapper, flag_combinations):
        # Execute constructor of the FeaturesExtractionMethod (Father Class)
        FeaturesExtractionMethod.__init__(self, raw_text, cleaned_text, preprocessed_text, tokens, class_of_interest, model_wrapper, flag_combinations)
        return

    def extract_features(self):
        self.feature_extraction_type = "SEN"

        # Split input text into a list of sentences
        sentences = self.sentences_splitter(self.raw_text)

        # Create one feature for each sentence
        features = []
        feature_id = 0
        position_count = 0  # Keep track of the position of the word in the input text
        for sentence in sentences:
            feature, position_count = self.fit_feature(feature_id,
                                                       sentence,
                                                       position_count)
            features.append(feature)
            feature_id += 1

        # If `flag_combinations` is `True` then create pairwise combination of features
        if self.flag_combinations is True:
            combination_features = self.combine_feature(features, 2, len(features), self.feature_extraction_type)
            features = features + combination_features

        return features

    def fit_feature(self, feature_id, sentence, current_word_position):
        position_count = current_word_position

        cleaned_sentence = self.model_wrapper.clean_function(sentence)

        sentence_tokens = self.model_wrapper.texts_to_tokens([cleaned_sentence])[0]

        positions_tokens = {}
        for token in sentence_tokens:
            positions_tokens[position_count] = token
            position_count += 1

        feature = Feature(feature_id,
                          self.feature_extraction_type,
                          self.create_description(feature_id),
                          positions_tokens)

        return feature, position_count

    @staticmethod
    def sentences_splitter(input_text):
        """ Split the input text into sentences.
        Args:    input_text (str): String containing the input text
        Returns: list(str) list of sentences
        """
        return nltk.sent_tokenize(input_text)

    @staticmethod
    def create_description(feature_id):
        return str(feature_id + 1) + "° Sentence"


class MultiLayerWordEmbeddingFeaturesExtraction(FeaturesExtractionMethod):
    """ MultiLayer Word Embedding Feature Extraction Class: Implementation of the FeaturesExtractionMethod Abstract Class. """

    def __init__(self, raw_text, cleaned_text, preprocessed_text, tokens, class_of_interest, embedding_tensor, model_wrapper, flag_combinations):
        FeaturesExtractionMethod.__init__(self, raw_text, cleaned_text, preprocessed_text, tokens, class_of_interest, model_wrapper, flag_combinations)
        self.embedding_matrix = embedding_tensor
        self.config = self.__read_configuration_file()
        self.mlwe_info = None
        return

    def extract_features(self):
        self.feature_extraction_type = "MLWE"
        features = None
        if self.config["clustering_alg"] == "kmeans":
            mlwe_unsupervised_analysis = KMeansEmbeddingUnsupervisedAnalysis(self.preprocessed_text, self.model_wrapper,
                                                                             self.embedding_matrix, self.tokens,
                                                                             self.config, self.class_of_interest)
            features = mlwe_unsupervised_analysis.extract_embedding_features()
            self.mlwe_info = mlwe_unsupervised_analysis.get_kmeans_info()

            if self.flag_combinations is True:
                combination_features = self.combine_feature(features, 2, len(features), self.feature_extraction_type)
                features = features + combination_features
        return features

    @staticmethod
    def __read_configuration_file():
        """ Reads the Multi-Layer Word Embedding configuration file.
        Returns:
            config (dict): dictionary with pos scheduled and tags for each pos
        """
        with open(r'config_files/mlwe_configuration.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config


class EmbeddingUnsupervisedAnalysis(ABC):

    def __init__(self, preprocessed_text, model_wrapper, embedding_matrix, tokens, config):
        self.preprocessed_text = preprocessed_text
        self.model_wrapper = model_wrapper
        self.embedding_matrix = embedding_matrix
        self.tokens = tokens
        self.config = config
        self.feature_extraction_type = "MLWE"
        return

    @abstractmethod
    def extract_embedding_features(self):
        return


class KMeansEmbeddingUnsupervisedAnalysis(EmbeddingUnsupervisedAnalysis):
    def __init__(self, preprocessed_text, model_wrapper, embedding_matrix, tokens, config, class_of_interest):
        EmbeddingUnsupervisedAnalysis.__init__(self, preprocessed_text, model_wrapper, embedding_matrix, tokens, config)
        self.__parse_kmeans_configuration()
        if self.standardization is True:
            self.__standardize_embedding_matrix()
        self.k_max = self.__k_max()
        self.k_clusters = {}
        self.kmeans_info = {}
        self.class_of_interest = class_of_interest
        return

    def extract_embedding_features(self):
        features = self.execute_kmeans_embedding_unsupervised_analysis()
        return features

    def execute_kmeans_embedding_unsupervised_analysis(self):
        features = []
        for k in range(2, self.k_max+1):
            clusters = self.__apply_kmeans(k)  # Perform K-Means with current k value (the list of k clusters is returned)
            k_features = self.__convert_clusters_to_features(clusters, k)

            features = features + k_features

        perturbation_manager_mlwe = pe.PerturbationManager(self.preprocessed_text,
                                                           self.tokens,
                                                           self.model_wrapper,
                                                           features,
                                                           flag_removal=True)

        perturbation_manager_mlwe.execute_perturbation_phase()

        perturbations_mlwe = perturbation_manager_mlwe.get_all_perturbations()

        local_explanation_manage_mlwe = le.LocalExplanationManager(self.preprocessed_text,
                                                                   self.model_wrapper,
                                                                   self.class_of_interest,
                                                                   perturbations_mlwe)

        local_explanations_mlwe, original_probabilities, original_label = local_explanation_manage_mlwe.execute_local_explanation_phase()

        top_k, most_informative_k = self.__search_most_informative_k_division(local_explanations_mlwe)
        features = most_informative_k["features"]
        self.kmeans_info["top_k"] = top_k
        self.kmeans_info["mlwe_local_explanations"] = local_explanations_mlwe
        self.kmeans_info["k_max"] = self.k_max
        self.kmeans_info["k_divisions"] = []
        for k, v in self.k_clusters.items():
            self.kmeans_info["k_divisions"].append(v)
        return features

    def __convert_clusters_to_features(self, clusters, k):
        features = []
        cluster_id = 0
        for cluster in clusters:
            feature = self.fit_feature(cluster_id, cluster, k, 1)
            features.append(feature)
            cluster_id += 1
        return features

    def fit_feature(self, feature_id, cluster, k, combination):
        positions_tokens = cluster

        feature = Feature(feature_id,
                          self.feature_extraction_type,
                          self.create_description(feature_id),
                          positions_tokens,
                          combination,
                          k)

        return feature

    @staticmethod
    def create_description(feature_id):
        return "Cluster {}".format(feature_id)

    def get_kmeans_info(self):
        return self.kmeans_info

    def __parse_kmeans_configuration(self):
        self.max_iterations = self.config["kmeans"]["max_iterations"]
        self.n_init = self.config["kmeans"]["n_init"]
        self.init_type = self.config["kmeans"]["init_type"]
        self.standardization = self.config["kmeans"]["standardization"]
        return

    def __k_max(self):
        """ The maximum number of K-partitions analyzed is defined as √(number_of_words+1).
        Returns:
            k_max (int): Max value for K
        """
        return int(math.sqrt(len(self.tokens) + 1))

    def __search_most_informative_k_division(self, local_explanations_mlwe):
        """ Evaluates each k division and finds the most informative one. """
        for local_explanation in local_explanations_mlwe:
            k = local_explanation.perturbation.feature.k

            if k not in self.k_clusters:
                self.k_clusters[k] = {"features": [local_explanation.perturbation.feature],
                                      "weighted_nPIRs": [local_explanation.numerical_explanation.nPIR_class_of_interest
                                                         / len(local_explanation.perturbation.feature.positions_tokens.keys())],
                                      "k": k}
            else:
                self.k_clusters[k]["features"].append(local_explanation.perturbation.feature)
                self.k_clusters[k]["weighted_nPIRs"].append(local_explanation.numerical_explanation.nPIR_class_of_interest
                                                            / len(local_explanation.perturbation.feature.positions_tokens.keys()))

        for k in self.k_clusters:
            self.k_clusters[k]["k_score"] = self.__k_score(self.k_clusters[k]["weighted_nPIRs"])

        top_k = max(self.k_clusters, key=lambda k: self.k_clusters[k]["k_score"])
        return top_k, self.k_clusters[top_k]

    def __apply_kmeans(self, k):
        """ Applies K-Means algorithm with the specified K over the tokens and the embedding matrix.

        Args:
            k (int): Number of clusters
        Return:
            clusters (list(dict)): List of dictionaries, each element of the list contains the cluster of words anche eache element of the dictionary is key=position, value=token
        """
        kmeans = KMeans(n_clusters=k, max_iter=self.max_iterations, n_init=self.n_init, init=self.init_type)
        kmeans.fit(self.embedding_matrix)
        lebels_word = kmeans.labels_

        clusters = []  # list of clusters, each cluster is a dictionary

        for cluster_index in range(k):
            cluster = {}
            for position in range(len(self.tokens)):
                if lebels_word[position] == cluster_index:
                    cluster[position] = self.tokens[position]
            clusters.append(cluster)

        return clusters

    @staticmethod
    def __k_score(nPIRs):
        max_nPIR = max(nPIRs)
        min_nPIR = min(nPIRs)
        return max_nPIR - min_nPIR

    def __standardize_embedding_matrix(self):
        mean = np.mean(self.embedding_matrix)
        st_dev = np.std(self.embedding_matrix)
        self.embedding_matrix = self.embedding_matrix - mean
        self.embedding_matrix = self.embedding_matrix / st_dev
        return

