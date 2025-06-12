#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/6/11 19:57
@desc: 
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset
import json


# ==== Load dataset from JSONL ====
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]
    return Dataset.from_list(data)


# ==== Model & Tokenizer ====
base_model_name = "Qwen/Qwen2.5-0.5B"  # You can replace with Qwen, LLaMA, etc.
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(base_model_name)
# print(model)

# ==== Apply LoRA ====
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],  # Use 'q_proj', 'v_proj' for LLaMA-like models
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)


# ==== Dynamic Prompt Construction ====
def construct_prompt(example):
    # Join few-shot examples
    few_shot = ""
    for fs in example["few_shot_examples"]:
        few_shot += f"Q: {fs['q']}\nA: {fs['a']}\n\n"
    prompt = few_shot + f"Q: {example['question']}\nA:"

    answer = example["answers"]["text"][0] if isinstance(example["answers"]["text"], list) else example["answers"][
        "text"]

    # Tokenize prompt and answer separately
    prompt_ids = tokenizer(prompt, truncation=True, max_length=384, padding="max_length")["input_ids"]
    answer_ids = tokenizer(answer, truncation=True, max_length=32, padding="max_length")["input_ids"]

    # Construct input and label: only compute loss on answer
    input_ids = prompt_ids[:-32] + answer_ids
    labels = [-100] * (len(prompt_ids) - 32) + answer_ids

    return {
        "input_ids": input_ids,
        "labels": labels
    }


# ==== Load and preprocess data ====
dataset = load_jsonl("few_shot_qa_dataset.jsonl")
processed_dataset = dataset.map(construct_prompt, remove_columns=dataset.column_names)

# ==== Training Setup ====
training_args = TrainingArguments(
    output_dir="./lora_qa_model",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

# ==== Trainer ====
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
)

# ==== Start Training ====
trainer.train()

# ==== Save adapter only ====
model.save_pretrained("./lora_qa_model")
tokenizer.save_pretrained("./lora_qa_model")

