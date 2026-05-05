import copy

from .base_recognizer import BaseRecognizer
from .distillation_recognizer import DistillationRecognizer

__all__ = ['build_model']


def build_model(config):
    config = copy.deepcopy(config)
    
    if config.get('name') == 'DistillationRecognizer':
        rec_model = DistillationRecognizer(config)
    else:
        rec_model = BaseRecognizer(config)
        
    return rec_model
