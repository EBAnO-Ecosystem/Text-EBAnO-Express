from collections import OrderedDict
import sys


class PerturbationScores:
    col_prediction = "prediction"
    col_prediction_t = "prediction_t"
    col_a_PIR = "a_PIR"
    col_b_PIR = "b_PIR"
    col_PIR = "PIR"
    col_nPIR = "nPIR"
    col_a_PIRP = "a_PIRP"
    col_b_PIRP = "b_PIRP"
    col_PIRP = "PIRP"
    col_nPIRP = "nPIRP"

    def __init__(self, P_o, P_t, coi):

        self.coi = coi

        self.P_o = P_o

        self.p_o = self.P_o[self.coi]

        if P_t is not None:
            self.P_t = P_t
            self.p_t = self.P_t[self.coi]
        else:
            self.P_t = None
            self.p_t = float('NaN')

        self.PIR = float('NaN')
        self.PIRP = float('NaN')

        self.nPIR = float('NaN')
        self.nPIRP = float('NaN')

        self.a_pir, self.b_pir = float('NaN'), float('NaN')
        self.classes_npir = float('NaN')
        self.w_c_npir = float('NaN')
        self.pirp_coi = float('NaN')
        self.pirp_no_coi = float('NaN')
        self.a_pirp, self.b_pirp = float('NaN'), float('NaN')

    @staticmethod
    def softsign_norm(x):
        x_n = x / (1 + abs(x))
        return x_n

    @staticmethod
    def relu(x):
        if x >= 0:
            return x
        else:
            return 0.0

    @staticmethod
    def _get_a_b(p_o, p_t):

        a = (1 - p_o / p_t)

        if a == float('inf'):
            a = sys.float_info.max
        elif a == float('-inf'):
            a = -sys.float_info.max

        b = (1 - p_t / p_o)

        if b == float('inf'):
            b = sys.float_info.max
            print("aaa")
        elif b == float('-inf'):
            b = -sys.float_info.max

        return a, b

    @staticmethod
    def compute_influence_relation(p_o, p_t):
        a, b = PerturbationScores._get_a_b(p_o, p_t)
        return (p_t * b) - (p_o * a)

    @staticmethod
    def compute_perturbation_influence_relation(p_o, p_t):
        return PerturbationScores.compute_influence_relation(p_o, p_t)

    @staticmethod
    def compute_perturbation_influence_relation_normalized(p_o, p_t):
        PIR = PerturbationScores.compute_perturbation_influence_relation(p_o, p_t)
        return PerturbationScores.softsign_norm(PIR)

    @staticmethod
    def compute_npir_for_all_classes(P_o, P_t):
        classes_npir = [PerturbationScores.compute_perturbation_influence_relation_normalized(p_o, p_t) for p_o, p_t in zip(P_o, P_t)]
        return classes_npir

    @staticmethod
    def weighted_classes_npir(classes_npir, weights):
        return classes_npir * weights

    @staticmethod
    def pirp_coi(w_c_npir, coi):
        pirp_coi = abs(w_c_npir[coi])
        return pirp_coi

    @staticmethod
    def pirp_no_coi(w_c_npir, coi):
        w_c_npir_no_coi = w_c_npir.copy()
        w_c_npir_no_coi[coi] = 0.0
        w_c_npir_no_coi = [PerturbationScores.relu(wir) for wir in w_c_npir_no_coi]
        pirp_no_coi = sum(w_c_npir_no_coi)
        return pirp_no_coi

    @staticmethod
    def compute_perturbation_influence_relation_precision(P_o, P_t, coi):
        classes_npir = PerturbationScores.compute_npir_for_all_classes(P_o, P_t)

        w_c_npir = PerturbationScores.weighted_classes_npir(classes_npir, P_o)

        pirp_coi = PerturbationScores.pirp_coi(w_c_npir, coi)
        pirp_no_coi = PerturbationScores.pirp_no_coi(w_c_npir, coi)

        return PerturbationScores.compute_influence_relation(pirp_coi, pirp_no_coi)

    @staticmethod
    def compute_perturbation_influence_relation_precision_normalized(P_o, P_t, coi):
        """
        se new_irp_simm > 0 -> la feature è precisa nella la classe in esame \n
        se new_irp_simm = 0 -> la feature non è precisa nella la classe in esame ma impatta anche altre classi \n
        se new_irp_simm < 0 -> la feature non è precisa nella la classe in esame e impatta maggiormente altre classi \n\n
        :param P_o:
        :param P_t:
        :param coi:
        :return:
        """
        pirp = PerturbationScores.compute_perturbation_influence_relation_precision(P_o, P_t, coi)
        return PerturbationScores.softsign_norm(pirp)

    def compute_scores(self):

        if self.P_t is None:
            return self

        self.PIR = PerturbationScores.compute_perturbation_influence_relation(self.p_o, self.p_t)
        self.nPIR = PerturbationScores.compute_perturbation_influence_relation_normalized(self.p_o, self.p_t)

        self.PIRP = PerturbationScores.compute_perturbation_influence_relation_precision(self.P_o, self.P_t, self.coi)
        self.nPIRP = PerturbationScores.compute_perturbation_influence_relation_precision_normalized(self.P_o, self.P_t, self.coi)

        self.a_pir, self.b_pir = PerturbationScores._get_a_b(self.p_o, self.p_t)

        self.classes_npir = PerturbationScores.compute_npir_for_all_classes(self.P_o, self.P_t)

        self.w_c_npir = PerturbationScores.weighted_classes_npir(self.classes_npir, self.P_o)

        self.pirp_coi = PerturbationScores.pirp_coi(self.w_c_npir, self.coi)
        self.pirp_no_coi = PerturbationScores.pirp_no_coi(self.w_c_npir, self.coi)

        self.a_pirp, self.b_pirp = PerturbationScores._get_a_b(self.pirp_coi, self.pirp_no_coi)

        return self

    def get_scores_dict(self):
        scores_dict = OrderedDict()

        scores_dict[self.col_prediction] = float(self.p_o)
        scores_dict[self.col_prediction_t] = float(self.p_t)

        # PIR - Perturbation Influence Relation
        scores_dict[self.col_a_PIR] = float(self.a_pir)
        scores_dict[self.col_b_PIR] = float(self.b_pir)
        scores_dict[self.col_PIR] = float(self.PIR)
        scores_dict[self.col_nPIR] = float(self.nPIR)

        # PIRP - Perturbation Influence Relation Precision
        scores_dict[self.col_a_PIRP] = float(self.a_pirp)
        scores_dict[self.col_b_PIRP] = float(self.b_pirp)
        scores_dict[self.col_PIRP] = float(self.PIRP)
        scores_dict[self.col_nPIRP] = float(self.nPIRP)

        return scores_dict

    def __str__(self):
        return str(self.get_scores_dict())
