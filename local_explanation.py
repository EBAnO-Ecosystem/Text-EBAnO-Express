from . import perturbation as pe
from . import feature_extraction as fe
from . import perturbation_scores
import numpy as np


class LocalExplanationManager:
    """ Local Explanation Phase Class.
    The LocalExplanationPhase class manages the workflow of the local explanation phase in the explanation process.

         Attributes:

        """
    def __init__(self, preprocessed_text, model_wrapper, class_of_interest, perturbations):
        """
        Args:
            preprocessed_text (str): String created by tokenizing and joining tokens (can contains <OOV> tokens)
            model_wrapper (:obj:ModelWrapperInterface)  Instance of a real class implementing the ModelWrapperInterface
            class_of_interest (int):
            perturbations (list[:obj:Perturbation]): List of perturbations to which create local explanations
        """
        self.input_text = preprocessed_text
        self.model_wrapper = model_wrapper
        self.class_of_interest = class_of_interest
        self.perturbations = perturbations
        self.original_probabilities = None
        self.original_label = None
        self.local_explanations = []
        return

    def execute_local_explanation_phase(self):
        """ Execute the Local Explanation Phase. """

        # Create a list of texts: first element is the original text, the others are the perturbed texts
        texts = [self.input_text]
        for perturbation in self.perturbations:
            texts.append(perturbation.perturbed_text)

        # Predict output probabilities for original text and perturbed texts
        predictions = self.model_wrapper.predict(texts)
        # Pop from the list the first element corresponding to the probabilities of the original text
        self.original_probabilities = predictions[0]
        self.original_label = self.model_wrapper.get_label_list()[np.argmax(self.original_probabilities)]

        # If the class_of_interest is `-1` then assign to it the max predicted class label from the original text
        if self.class_of_interest == -1:
            coi = self.original_label
        else:
            coi = self.class_of_interest

        local_explanation_id = 0
        for prediction, perturbation in zip(predictions[1:], self.perturbations):

            perturbed_probabilities = prediction

            try:
                # Compute the scores for the `class_of_interest` -> How much the current feature is influential for the `class_of_interest`
                ps_coi = perturbation_scores.PerturbationScores(self.original_probabilities, perturbed_probabilities, coi)
                ps_coi.compute_scores()

                nPIR_class_of_interest = ps_coi.nPIR
                nPIRP_class_of_interest = ps_coi.nPIRP
            except:
                print("An exception has occurred while calculating nPIR and nPIRP.")
                nPIR_class_of_interest = round(0, 3)
                nPIRP_class_of_interest = round(1, 3)

            # original_top_class is the most likely predicted label on the Original text
            original_top_class = self.model_wrapper.get_label_list()[np.argmax(self.original_probabilities)]

            # perturbed_top_class is the most likely predicted label on the Perturbed text
            perturbed_top_class = self.model_wrapper.get_label_list()[np.argmax(perturbed_probabilities)]

            # Check if the original predicted class is different from the class of interest
            if original_top_class != coi:
                try:
                    # Compute the scores for the `original_top_class` -> How much the current feature is influential for the `original_top_class`
                    ps_original_top_class = perturbation_scores.PerturbationScores(self.original_probabilities, perturbed_probabilities,
                                                                                   original_top_class)
                    ps_original_top_class.compute_scores()

                    nPIR_original_top_class = ps_original_top_class.nPIR
                    nPIRP_original_top_class = ps_original_top_class.nPIRP
                except:
                    print("An exception has occurred while calculating nPIR and nPIRP.")
                    nPIR_original_top_class = round(0, 3)
                    nPIRP_original_top_class = round(1, 3)
            else:
                # If the original top class is the same of the class of interests, then the scores are already computed
                nPIR_original_top_class = nPIR_class_of_interest
                nPIRP_original_top_class = nPIRP_class_of_interest

            nPIRs = []
            nPIRPs = []
            # Calculate the scores for each possible class label -> How much the current feature is influential for each class label
            for class_label in self.model_wrapper.get_label_list():
                try:
                    ps = perturbation_scores.PerturbationScores(self.original_probabilities, perturbed_probabilities, class_label)
                    ps.compute_scores()

                    nPIR = ps.nPIR
                    nPIRP = ps.nPIRP
                except:
                    print("An exception has occurred while calculating nPIR and nPIRP.")
                    nPIR = round(0, 3)
                    nPIRP = round(1, 3)

                nPIRs.append(nPIR)
                nPIRPs.append(nPIRP)

            # Create an instance of the NumericalExplanation
            numerical_explanation = NumericalExplanation(nPIR_original_top_class,  # nPIR obtained by the current feature on the original predicted class
                                                         nPIRP_original_top_class,  # nPIRP obtained by the current feature on the original predicted class
                                                         nPIR_class_of_interest,  # nPIR obtained by the current feature on the class of interest
                                                         nPIRP_class_of_interest,  # nPIRP obtained by the current feature on the class of interest
                                                         nPIRs,  # List of all nPIR obtained by the current feature on each class label
                                                         nPIRs)  # List of all nPIRP obtained by the current feature on each class label

            # Create an instance of the LocalExplanation
            local_explanation = LocalExplanation()

            local_explanation.fit(local_explanation_id,  # Unique identifier of the local explanation for each input text
                                  perturbation,  # Perturbation instance (each perturbation contains a single Feature instance)
                                  self.original_probabilities.tolist(),  # probabilities predicted by the model on the original text
                                  perturbed_probabilities.tolist(),  # probabilities predicted by the model on the perturbed version of text
                                  original_top_class,  # Most likely class predicted by the model on the original text
                                  perturbed_top_class,   # Most likely class predicted by the model on the perturbed text
                                  coi,  # Class of interest
                                  numerical_explanation)  # Instance of the numerical explanation of the current feature

            self.local_explanations.append(local_explanation)
            local_explanation_id += 1

        return self.local_explanations, self.original_probabilities, self.original_label


class LocalExplanation:
    def __init__(self):
        self.local_explanation_id = None
        self.perturbation = None
        self.original_probabilities = None
        self.perturbed_probabilities = None
        self.original_top_class = None
        self.perturbed_top_class = None
        self.class_of_interest = None
        self.numerical_explanation = None
        return

    def fit(self, local_explanation_id, perturbation, original_probabilities, perturbed_probabilities,
            original_top_class, perturbed_top_class, class_of_interest, numerical_explanation):
        self.local_explanation_id = local_explanation_id
        self.perturbation = perturbation
        self.original_probabilities = original_probabilities
        self.perturbed_probabilities = perturbed_probabilities
        self.original_top_class = original_top_class
        self.perturbed_top_class = perturbed_top_class
        self.class_of_interest = class_of_interest
        self.numerical_explanation = numerical_explanation
        return

    def fit_from_dict(self, local_explanation_dict):
        self.local_explanation_id = local_explanation_dict["local_explanation_id"]
        self.original_probabilities = local_explanation_dict["original_probabilities"]
        self.perturbed_probabilities = local_explanation_dict["perturbed_probabilities"]
        self.original_top_class = local_explanation_dict["original_top_class"]
        self.perturbed_top_class = local_explanation_dict["perturbed_top_class"]
        self.class_of_interest = local_explanation_dict["class_of_interest"]

        feature = fe.Feature(local_explanation_dict["feature_id"],
                             local_explanation_dict["feature_type"],
                             local_explanation_dict["feature_description"],
                             local_explanation_dict["positions_tokens"],
                             local_explanation_dict["combination"])

        self.perturbation = pe.Perturbation(local_explanation_dict["perturbation_id"],
                                            local_explanation_dict["perturbation_type"],
                                            local_explanation_dict["perturbed_text"],
                                            feature)

        self.numerical_explanation = NumericalExplanation(local_explanation_dict["nPIR_original_top_class"],
                                                          local_explanation_dict["nPIRP_original_top_class"],
                                                          local_explanation_dict["nPIR_class_of_interest"],
                                                          local_explanation_dict["nPIRP_class_of_interest"],
                                                          local_explanation_dict["nPIRs"],
                                                          local_explanation_dict["nPIRPs"])

        return

    def local_explanation_to_dict(self):
        """ Converts a single local explanation into dictionary. """
        perturbation = self.perturbation
        feature = perturbation.feature
        local_explanation_dict = {
            "local_explanation_id": self.local_explanation_id,
            "feature_id": feature.feature_id,
            "feature_type": feature.feature_type,
            "feature_description": feature.description,
            "positions_tokens": feature.positions_tokens,
            "combination": feature.combination,
            "perturbation_id": perturbation.perturbation_id,
            "perturbation_type": perturbation.perturbation_type,
            "perturbed_text": perturbation.perturbed_text,
            "original_probabilities": self.original_probabilities,
            "perturbed_probabilities": self.perturbed_probabilities,
            "original_top_class": self.original_top_class,
            "perturbed_top_class": self.perturbed_top_class,
            "class_of_interest": self.class_of_interest,
            "nPIR_original_top_class": self.numerical_explanation.nPIR_original_top_class,
            "nPIRP_original_top_class": self.numerical_explanation.nPIRP_original_top_class,
            "nPIR_class_of_interest": self.numerical_explanation.nPIR_class_of_interest,
            "nPIRP_class_of_interest": self.numerical_explanation.nPIRP_class_of_interest,
            "nPIRs": self.numerical_explanation.nPIRs,
            "nPIRPs": self.numerical_explanation.nPIRPs,
            "k": self.perturbation.feature.k
        }
        return local_explanation_dict


class NumericalExplanation:
    def __init__(self, nPIR_original_top_class, nPIRP_original_top_class, nPIR_class_of_interest,
                 nPIRP_class_of_interest, nPIRs, nPIRPs):
        self.nPIR_original_top_class = nPIR_original_top_class
        self.nPIRP_original_top_class = nPIRP_original_top_class
        self.nPIR_class_of_interest = nPIR_class_of_interest
        self.nPIRP_class_of_interest = nPIRP_class_of_interest
        self.nPIRs = nPIRs
        self.nPIRPs = nPIRPs
        return

