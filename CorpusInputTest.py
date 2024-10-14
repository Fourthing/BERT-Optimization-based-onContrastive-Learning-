from datasets import load_dataset

dataset = load_dataset("glue", "mrpc", split="train")

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def encode(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")

dataset = dataset.map(encode, batched=True)
dataset[0]

dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)