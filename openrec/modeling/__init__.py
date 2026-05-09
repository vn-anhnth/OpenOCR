import copy

from .base_recognizer import BaseRecognizer

__all__ = ['build_model']


def build_model(config):
    config = copy.deepcopy(config)
    if 'models' in config:
        from .distillation_model import DistillationModel
        rec_model = DistillationModel(config)
    else:
        rec_model = BaseRecognizer(config)
    return rec_model
