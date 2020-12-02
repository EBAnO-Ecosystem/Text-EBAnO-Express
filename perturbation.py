from abc import ABC, abstractmethod
from typing import List

from keras.preprocessing.text import text_to_word_sequence
import nltk
import re


class PerturbationManager:
    """ Perturbation Manager CLass.
    The PerturbationPhase class manages the workflow of the perturbation phase in the explanation process.
    Actually it is implemented only the removal perturbation.

    Attributes:
    """
    def __init__(self, preprocessed_text, tokens, model_wrapper, features, flag_removal=True):
        """ PerturbationPhase Initializer.
        Args:
            preprocessed_text (str): String created by tokenizing and joining tokens (can contains <OOV> tokens)
            model_wrapper (:obj:ModelWrapperInterface)  Instance of a real class implementing the ModelWrapperInterface
            features (List[:obj:Feature]): List of features to which apply perturbations
            flag_removal (bool): True if want apply removal perturbation, False otherwise
        """
        self.preprocessed_text = preprocessed_text
        self.tokens = tokens
        self.model_wrapper = model_wrapper
        self.features = features
        self.flag_removal = flag_removal
        self.removal_perturbation_method = None
        self.removal_perturbations = []

    def execute_perturbation_phase(self):
        """ Execute the Feature Extraction Phase. """
        if self.flag_removal:
            # If flag_removal is True, then instantiate the RemovalPerturbation and apply perturbation to each feature
            self.removal_perturbation_method = RemovalPerturbation(self.preprocessed_text, self.tokens, self.features)

            self.removal_perturbations = self.removal_perturbation_method.apply_perturbations()
        return

    def get_removal_perturbations(self):
        """ Returns: (list[Perturbation]) List containing all Removal perturbations. """
        return self.removal_perturbations

    def get_all_perturbations(self):
        """ Returns: (list[Perturbation]) List containing all the perturbations performed. """
        return self.removal_perturbations


class Perturbation:
    """ Perturbation Class.

    Perturbation represents a perturbation applied on the input text over a single feature extracted.
    """
    def __init__(self, perturbation_id, perturbation_type, perturbed_text, feature):
        """ Feature Initializer.
        Args:
            perturbation_id (int): Perturbation identifier
            perturbation_type (str): Perturbation method used
            perturbed_text (str): New text produced after the perturbation
            feature (Feature): Feature on which is applied the perturbation
        """
        self.perturbation_id = perturbation_id
        self.feature = feature
        self.perturbation_type = perturbation_type
        self.perturbed_text = perturbed_text

    def print_perturbation_info(self):
        """ Print information about the Perturbation. """
        print("Perturbation ID: ", self.perturbation_id)
        print("Perturbation Type: ", self.perturbation_type)
        print("Perturbed Text: ", self.perturbed_text)
        print("---")
        self.feature.print_feature_info()
        print("---")

    def get_perturbation_id(self):
        """ Returns: (int) perturbation id. """
        return self.perturbation_id

    def get_perturbation_method(self):
        """ Returns: (str) perturbation method used. """
        return self.perturbation_type

    def get_perturbed_text(self):
        """ Returns: (str) new text produced after the perturbation. """
        return self.perturbed_text

    def get_feature(self):
        """ Returns: (Feature) feature on which was applied the perturbation. """
        return self.feature


class PerturbationMethod(ABC):
    """ Abstract Class: Perturbation Method. """
    def __init__(self, preprocessed_text, tokens, features):
        """
        Args:
            preprocessed_text (str): String containing the input text
            features (list[Feature]): List of extracted features
        """
        self.preprocessed_text = preprocessed_text
        self.tokens = tokens
        self.features = features
        self.perturbation_type = None
        super().__init__()

    @abstractmethod
    def apply_perturbations(self) -> List[Perturbation]:
        """ Abstract Method: Apply the Perturbation to each Extracted Feature. """
        pass


class RemovalPerturbation(PerturbationMethod):
    """ Removal Perturbation Class: Implementation of the PerturbationMethod Abstract Class.

    The Removal Perturbation method produces a new Perturbed Text for each extracted feature by removing the tokens of the feature (separately for each feature).
    """
    def __init__(self, preprocessed_text, tokens, features):
        PerturbationMethod.__init__(self, preprocessed_text, tokens, features)
        return

    def apply_perturbations(self):
        """ Applies the removal perturbation to each feature passed in the constructor and produces a list of Perturbations. """
        self.perturbation_type = "Removal Perturbation"

        perturbations = []
        perturbation_id = 0
        # Loop over each feature
        for feature in self.features:
            # Create a perturbed text by removing each token of the feature from the original list of tokens
            perturbed_text = self.create_perturbed_text(self.tokens, feature)

            # create a Perturbation instance
            perturbation = self.fit_perturbation(perturbation_id, feature, perturbed_text)
            perturbations.append(perturbation)

            #perturbation.print_perturbation_info()
            perturbation_id += 1

        return perturbations

    def fit_perturbation(self, perturbation_id, feature, perturbed_text):
        """ Create a single Perturbation instance. """
        perturbation = Perturbation(perturbation_id,
                                    self.perturbation_type,
                                    perturbed_text,
                                    feature,)
        return perturbation

    def create_perturbed_text(self, tokens, feature):
        """ Create a perturbed version of text by removing the tokens of the feature from the original tokens.  """
        # Create a list of tokens by removing from the original list of tokens the ones belonging the feature
        perturbed_tokens = {i: tokens[i] for i in range(len(tokens)) if i not in feature.position_word.keys()}

        # Construct the sentence by joining the words
        perturbed_text = " ".join([token for token in perturbed_tokens.values()])

        # Apply a post-processing function to the perturbed version of the input text
        perturbed_text = self.__perturbed_text_post_processing(perturbed_text)
        return perturbed_text

    @staticmethod
    def __perturbed_text_post_processing(perturbed_text):
        """ Post-processing for the perturbed version of text. """
        # Compact multiple blank spaces to one space
        perturbed_text = re.sub("\s+", " ", perturbed_text.strip())

        # Remove blank space Before Punctuation
        perturbed_text = re.sub(r'\s+([?.!",;:])', r'\1', perturbed_text)
        return perturbed_text



