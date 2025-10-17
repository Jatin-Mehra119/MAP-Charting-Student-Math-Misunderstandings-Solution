from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
TEMP_DIR = "tmp"
os.makedirs(TEMP_DIR, exist_ok=True)
train = pd.read_csv('/content/drive/MyDrive/map-charting-student-math-misunderstandings/train.csv')

# Fill missing Misconception values with 'NA'
train.Misconception = train.Misconception.fillna('NA')

# Create a combined target label (Category:Misconception)
train['target'] = train.Category + ":" + train.Misconception

# Encode target labels to numerical format
le = LabelEncoder()
train['label'] = le.fit_transform(train['target'])
n_classes = len(le.classes_) # Number of unique target classes:65
print(f"Train shape: {train.shape} with {n_classes} target classes")
print(train.head())

idx = train.apply(lambda row: row.Category.split('_')[0], axis=1) == 'True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId', 'MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c', ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId', 'MC_Answer']]
correct['is_correct'] = 1 # Mark these as correct answers

# Merge 'is_correct' flag into the main training DataFrame
train = train.merge(correct, on=['QuestionId', 'MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)
from IPython.display import display, Math, Latex
tmp = train.groupby(['QuestionId','MC_Answer']).size().reset_index(name='count')
tmp['rank'] = tmp.groupby('QuestionId')['count'].rank(method='dense', ascending=False).astype(int) - 1
tmp = tmp.drop('count',axis=1)
tmp = tmp.sort_values(['QuestionId','rank'])

Q = tmp.QuestionId.unique()
for q in Q:
    question = train.loc[train.QuestionId==q].iloc[0].QuestionText
    choices = tmp.loc[tmp.QuestionId==q].MC_Answer.values
    labels="ABCD"
    choice_str = " ".join([f"({labels[i]}) {choice}" for i, choice in enumerate(choices)])

    print()
    display(Latex(f"QuestionId {q}: {question}") )
    display(Latex(f"MC Answers: {choice_str}"))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

Model_Name = "deepseek-ai/deepseek-math-7b-instruct"

model = AutoModelForSequenceClassification.from_pretrained(Model_Name, num_labels=n_classes,
                                                           torch_dtype=torch.bfloat16, device_map="balanced",
                                                          cache_dir=TEMP_DIR)

tokenizer = AutoTokenizer.from_pretrained(Model_Name, cache_dir=TEMP_DIR)


def format_input(row):
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return (
        f"Question: {row['QuestionText']}\n"
        f"Answer: {row['MC_Answer']}\n"
        f"Correct? {x}\n"
        f"Student Explanation: {row['StudentExplanation']}"
    )

train['text'] = train.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0] )

from datasets import Dataset

# Convert to Hugging Face Dataset
COLS = ['text', 'label']

# Create clean DataFrame with the full training data
train_df_clean = train[COLS].copy()  # Use 'train' instead of 'train_df'

# Ensure labels are proper integers
train_df_clean['label'] = train_df_clean['label'].astype(np.int64)

# Reset index to ensure clean DataFrame structure
train_df_clean = train_df_clean.reset_index(drop=True)

# Create dataset with the full training data
train_ds = Dataset.from_pandas(train_df_clean, preserve_index=False)
def tokenize(batch):
    """Tokenizes a batch of text inputs."""
    return tokenizer(batch["text"], truncation=True, max_length=256)

# Apply tokenization to the full dataset
train_ds = train_ds.map(tokenize, batched=True, remove_columns=['text'])

# Add a new padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# Resize the model's token embeddings to match the new tokenizer
model.resize_token_embeddings(len(tokenizer))
# Set the pad token id in the model's config
model.config.pad_token_id = tokenizer.pad_token_id

# --- Training Arguments ---
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

# Ensure temp directories exist
os.makedirs(f"{TEMP_DIR}/training_output/", exist_ok=True)
os.makedirs(f"{TEMP_DIR}/logs/", exist_ok=True)

# --- Training Arguments (FIXED) ---
training_args = TrainingArguments(
    output_dir=f"{TEMP_DIR}/training_output/",  
    do_train=True,
    do_eval=False,
    save_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    logging_dir=f"{TEMP_DIR}/logs/",           
    logging_steps=500,
    bf16=True,
    fp16=False,
    report_to="none",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    dataloader_pin_memory=False,
    gradient_checkpointing=True,              # Added: Helps with memory
)
# --- Custom Metric Computation (MAP@3) ---
def compute_map3(eval_pred):
    """
    Computes Mean Average Precision at 3 (MAP@3) for evaluation.
    """
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    # Get top 3 predicted class indices for each sample
    top3 = np.argsort(-probs, axis=1)[:, :3]

    # Check if the true label is within the top 3 predictions
    match = (top3 == labels[:, None]) # Create a boolean array indicating matches

    map3 = 0.0
    for i in range(len(labels)):
        if match[i, 0]: # If true label is in the 1st prediction
            map3 += 1.0
        elif match[i, 1]: # If true label is in the 2nd prediction
            map3 += 1.0 / 2
        elif match[i, 2]: # If true label is in the 3rd prediction
            map3 += 1.0 / 3

    return {"map@3": map3 / len(labels)} # Average MAP@3 over all samples

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create trainer without evaluation dataset and metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# Start training on the full dataset
trainer.train()

import joblib

# --- Save Model and Label Encoder ---
print(f"Saving best model to best")
trainer.save_model(f"/content/drive/MyDrive/map-charting-student-math-misunderstandings/deepseekmaths/best") 
joblib.dump(le, f"/content/drive/MyDrive/map-charting-student-math-misunderstandings/deepseekmaths/label_encoder.joblib")