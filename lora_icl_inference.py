#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/6/11 19:58
@desc: 
"""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList

from stop_word import StopOnNewline

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
model = PeftModel.from_pretrained(base_model, "./lora_qa_model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token

prompt = """Q: What is the capital of Germany?
A: Berlin

Q: Who painted the Mona Lisa?
A: Leonardo da Vinci

Q: What is the largest ocean?
A:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
ipt_len = inputs["input_ids"].shape[-1]
stopping_criteria = StoppingCriteriaList([StopOnNewline(tokenizer, start_len=ipt_len)])
outputs = model.generate(**inputs, max_new_tokens=5,
                         pad_token_id=tokenizer.eos_token_id,
                         stopping_criteria=stopping_criteria,
                         )
print(tokenizer.decode(outputs[0][ipt_len:], skip_special_tokens=True))

