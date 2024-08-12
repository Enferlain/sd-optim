import sd_mecha
from sd_mecha.hypers import Hyper

class MergeMethods:
    @staticmethod
    def weighted_sum(a, b, alpha: Hyper):
        return sd_mecha.weighted_sum(a, b, alpha=alpha)

    @staticmethod
    def rotate(a, b, alpha: Hyper, beta: Hyper):
        return sd_mecha.rotate(a, b, alignment=alpha, alpha=beta)
