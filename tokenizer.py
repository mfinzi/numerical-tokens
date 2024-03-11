from plum import dispatch
import numpy as np
import torch

from collections import namedtuple, OrderedDict
from dataclasses import dataclass
import random
#assumes using transformers tokenizers

@dataclass
class TokenizerSettings:
    base_tokenizer: callable
    decimal_precision: int = 3 # number of decimals to keep for floats
    scaler: float = 1. # scaling for floating point numbers
    decimal_half_bin_correction: bool = False
    separator: str = ','
    random_transform: bool = False
    #dequantization_noise: bool = False

# default_settings = TokenizerSettings()

@dispatch
def tokenize(obj: str, settings):
    return settings.base_tokenizer(obj, add_special_tokens=False)['input_ids']

@dispatch
def tokenize(obj: list | tuple | np.ndarray | torch.Tensor, settings):
    if isinstance(obj, (np.ndarray, torch.Tensor)) and obj.ndim == 0:
        return tokenize(obj.item(), settings)
    out = []
    for i, item in enumerate(obj):
        if i > 0:
            out.append(settings.base_tokenizer.convert_tokens_to_ids(settings.separator))
        out.extend(tokenize(item, settings))
    return out

@dispatch
def tokenize(obj: float, settings):
    number_string = f"{obj/settings.scaler:.{settings.decimal_precision}f}"
    return settings.base_tokenizer.convert_tokens_to_ids(list(number_string))

@dispatch
def tokenize(obj: int, settings):
    return settings.base_tokenizer.convert_tokens_to_ids(list(str(obj)))

@dispatch
def tokenize(obj: set, settings):
    out = []
    items = list(obj)
    if settings.random_transform:
        random.shuffle(items)  # apply a random permutation
    out.append(settings.base_tokenizer.convert_tokens_to_ids('{'))
    for v in items:
        if len(out) > 0:
            out.append(settings.base_tokenizer.convert_tokens_to_ids(settings.separator))
        out.extend(tokenize(v, settings))
    out.append(settings.base_tokenizer.convert_tokens_to_ids('}'))
    return out

@dispatch
def tokenize(obj: dict, settings):
    out = []
    items = list(obj.items())
    if settings.random_transform and not isinstance(obj, OrderedDict):
        random.shuffle(items)  # apply a random permutation
    out.append(settings.base_tokenizer.convert_tokens_to_ids('{'))
    for k, v in items:
        if len(out) > 0:
            out.append(settings.base_tokenizer.convert_tokens_to_ids(settings.separator))
        out.extend(tokenize(k, settings))+tokenize(obj.x,s)+tokenize(',',s)
        out.append(settings.base_tokenizer.convert_tokens_to_ids(':'))
        out.extend(tokenize(v, settings))
    out.append(settings.base_tokenizer.convert_tokens_to_ids('}'))
    return out


Unitful = namedtuple('Unitful', ['value', 'unit'])

@dispatch
def tokenize(obj: Unitful, settings):
    value_tokens = tokenize(obj.value, settings)
    unit_tokens = tokenize(obj.unit, settings)
    return value_tokens + unit_tokens


# class Unitful(object):
#     def __init__(self, value, unit):
#         self.value = value
#         self.unit = unit

# a different way of doing it if we want
# class Decimal(object):
#     def __init__(self, value, digits=3, scaler=1.):
#         """ digits specifies number of digits after the decimal place. Scaler scales the value"""
#         self.value = value
#         self.digits = digits
#         self.scaler = scaler

# @dispatch
# def tokenize(obj: Decimal, base_tokenizer, settings):
#     number_string = f"{obj.value/obj.scaler:.{obj.digits}f}"
#     return base_tokenizer.convert_tokens_to_ids(list(number_string))

