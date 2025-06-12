#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/6/12 9:25
@desc: 
"""
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnNewline(StoppingCriteria):
    def __init__(self, tokenizer, start_len):
        self.tokenizer = tokenizer
        self.start_len = start_len

    def __call__(self, input_ids, scores, **kwargs):
        new_tokens = input_ids[0][self.start_len:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return '\n' in decoded





