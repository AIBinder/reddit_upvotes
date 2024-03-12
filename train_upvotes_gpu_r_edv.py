import pandas as pd 
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import evaluate
import torch

# optionally for debugging CUDA errors
#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_multiple_json_objects(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data

data = load_multiple_json_objects("de_EDV_submissions")

# example submission
data_ex = data[11]
print("title: " + data_ex['title'])
print("author: " + data_ex['author'])
print("text: " + data_ex['selftext'])
print("score: " + str(data_ex['score']))

# get score distribution
scores = []
for subm in data:
    #grouped scores in intervals 0-1,2-5,6-10,11-30,31-2000
    if subm['score'] <= 1:
        subm['score_gr'] = 0
    elif subm['score'] <= 5:
        subm['score_gr'] = 1
    elif subm['score'] <= 10:
        subm['score_gr'] = 2
    elif subm['score'] <= 29:
        subm['score_gr'] = 3
    else:
        subm['score_gr'] = 4
    scores.append(subm['score_gr'])
scores = pd.Series(scores)
# get distribution with occurences of each score, ordered by score
score_dist = scores.value_counts().sort_index()
#group in intervals 0-1,2-5,6-10,11-29,30+
print(score_dist)

# prepare data for LLM-finetuning
# create dataframe from data
data_processed = {}
data_processed["train"]=[]
data_processed["test"]=[]
print(len(data))
np.random.shuffle(data) # Todo: export data prep/split to utils function (and set seed to match partition in eval script)
for i,subm in enumerate(data):
    subm_proc = {}
    subm_proc['text'] = "title: " + subm['title'] + " \n " + "author: " + subm['author'] +  " \n " +  "text: " + subm['selftext']
    subm_proc['label'] = subm['score_gr']
    #randomly split data into train and test (80/20)
    if i < 0.8*len(data):
        data_processed["train"].append(subm_proc)
    else:
        data_processed["test"].append(subm_proc)

df = {}
df["train"] = pd.DataFrame(data_processed["train"])
df["test"] = pd.DataFrame(data_processed["test"])

#sample 1000 examples for each score group in df["train"]
df["train"] = df["train"].groupby('label').apply(lambda x: x.sample(1000)).reset_index(drop=True)
df["test"] = df["test"].groupby('label').apply(lambda x: x.sample(200)).reset_index(drop=True)

#show first rows
print(df["train"].head())

datasets = {}
datasets["train"] = Dataset.from_pandas(df["train"])
datasets["test"] = Dataset.from_pandas(df["test"])

# Check if a GPU is available for training and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load deepset gbert-large, should be suited for short German texts
tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "deepset/gbert-base",
    num_labels=5)
model.to(device)

def tokenize_and_pad_function(examples):
    return tokenizer(examples["text"], max_length=512, truncation=True, padding="max_length")

tokenized_datasets = {}
tokenized_datasets["train"] = datasets["train"].map(tokenize_and_pad_function, batched=True)
tokenized_datasets["test"] = datasets["test"].map(tokenize_and_pad_function, batched=True)

print(tokenized_datasets["train"][0])

# prepare for training and eval
training_args = TrainingArguments(
        output_dir="test_trainer",                          # Output directory
        num_train_epochs=3,              # Total number of training epochs
        per_device_train_batch_size=8,  # Batch size per device during training
        per_device_eval_batch_size=8,                   # Batch size for evaluation
        warmup_steps=20,                               # Number of warmup steps for learning rate scheduler
        #learning_rate=5e-5,
        #weight_decay=0.01,                              # Strength of weight decay
        logging_dir='./logs',                           # Directory for storing logs
        #logging_steps=1,
        evaluation_strategy="epoch",                    # Evaluation is done (and logged) every logging_steps steps.
        save_strategy="epoch",                          # Model checkpointing is also done every logging_steps steps.
        #save_steps=1,
        load_best_model_at_end=True,                    # Load the best model at the end of training
    )

# alternativ mit hyperparameter tuning später
#training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# train parameters
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(3000)), #subset for dev
    eval_dataset=tokenized_datasets["test"].shuffle(seed=42),#.select(range(700)),
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()