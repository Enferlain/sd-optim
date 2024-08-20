import sd_mecha
from sd_mecha.hypers import Hyper


class MergeMethods:
    @staticmethod
    def weighted_sum(a, b, alpha: Hyper):
        return sd_mecha.weighted_sum(a, b, alpha=alpha)

    @staticmethod
    def rotate(a, b, alignment: Hyper, alpha: Hyper):
        return sd_mecha.rotate(a, b, alignment=alignment, alpha=alpha)

    @staticmethod
    def tensor_sum(a, b, width: Hyper, offset: Hyper):
        return sd_mecha.tensor_sum(a, b, width=width, offset=offset)
