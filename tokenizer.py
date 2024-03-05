from plum import dispatch
import numpy as np
import torch

#from collections import namedtuple
from dataclasses import dataclass

#assumes using transformers tokenizers

@dataclass
class TokenizerSettings:
    decimal_precision: int = 3 # number of decimals to keep for floats
    scaler: float = 1. # scaling for floating point numbers
    decimal_half_bin_correction: bool = False
    separator: str = ','
    #dequantization_noise: bool = False

default_settings = TokenizerSettings()

@dispatch
def tokenize(obj: str, base_tokenizer, settings=default_settings):
    return base_tokenizer(obj, add_special_tokens=False)['input_ids']

@dispatch
def tokenize(obj: list | tuple | np.ndarray | torch.Tensor, base_tokenizer, settings=default_settings):
    out = []
    for i, item in enumerate(obj):
        if i > 0:
            out.append(base_tokenizer.convert_tokens_to_ids(settings.separator))
        out.extend(tokenize(item, base_tokenizer))
    return out

@dispatch
def tokenize(obj: float, base_tokenizer, settings=default_settings):
    number_string = f"{obj/settings.scaler:.{settings.decimal_precision}f}"
    return base_tokenizer.convert_tokens_to_ids(list(number_string))

@dispatch
def tokenize(obj: int, base_tokenizer, settings=default_settings):
    return base_tokenizer.convert_tokens_to_ids(list(str(obj)))

# a different way of doing it if we want
# class Decimal(object):
#     def __init__(self, value, digits=3, scaler=1.):
#         """ digits specifies number of digits after the decimal place. Scaler scales the value"""
#         self.value = value
#         self.digits = digits
#         self.scaler = scaler

# @dispatch
# def tokenize(obj: Decimal, base_tokenizer, settings=default_settings):
#     number_string = f"{obj.value/obj.scaler:.{obj.digits}f}"
#     return base_tokenizer.convert_tokens_to_ids(list(number_string))

