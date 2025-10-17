import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import pandas as pd
import torch
from argparse import Namespace
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, args):
        self.args = args
        self.le = None
        self.isPreprocess = False
        self.correct_lookup = None

    def load_data(self):
        self.train_df = pd.read_csv(self.args.train_path)
        self.test_df = pd.read_csv(self.args.test_path)

    def get_num_classes(self):
        if self.isPreprocess == False:
            return "please preprocess first"
        num_class = self.train_df['label'].nunique()
        return num_class

    def get_label_encoder(self):
        if self.le is None:
            raise ValueError("LabelEncoder not initialized. Please run preprocess first.")
        return self.le

    @staticmethod
    def format_input(row):
        correct_text = "Yes" if row['IsCorrect'] else "No"
        return (
            f"Question: {row['QuestionText']}\n"
            f"Answer: {row['MC_Answer']}\n"
            f"Correct? {correct_text}\n"
            f"Student Explanation: {row['StudentExplanation']}\n"
        )

    def preprocess(self):
        self.load_data()
        self.train_df['Misconception'] = self.train_df['Misconception'].fillna('NA')
        self.train_df['target'] = self.train_df['Category'] + ':' + self.train_df['Misconception']

        correct_samples = self.train_df[self.train_df['Category'].str.startswith('True', na=False)].copy()
        correct_samples['count'] = correct_samples.groupby(['QuestionId', 'MC_Answer'])['MC_Answer'].transform('count')
        most_popular_correct = correct_samples.sort_values('count', ascending=False).drop_duplicates(['QuestionId'])
        self.correct_lookup = most_popular_correct[['QuestionId', 'MC_Answer']].copy()
        self.correct_lookup['IsCorrect_flag'] = True

        self.train_df = self.train_df.merge(self.correct_lookup, on=['QuestionId', 'MC_Answer'], how='left')
        self.train_df['IsCorrect'] = self.train_df['IsCorrect_flag'].notna()
        self.train_df = self.train_df.drop(columns=['IsCorrect_flag'])

        self.le = LabelEncoder()
        self.train_df['label'] = self.le.fit_transform(self.train_df['target'])
        self.train_df['text'] = self.train_df.apply(self.format_input, axis=1)

        self.isPreprocess = True
        return self.train_df

    def inference_processor(self):
        if self.isPreprocess == False:
            return "Have you do the train? please preprocess first"
        self.test_df = self.test_df.merge(self.correct_lookup, on=['QuestionId', 'MC_Answer'], how='left')
        self.test_df['IsCorrect'] = self.test_df['IsCorrect_flag'].notna()
        self.test_df = self.test_df.drop(columns=['IsCorrect_flag'])
        self.test_df['text'] = self.test_df.apply(self.format_input, axis=1)
        return self.test_df

# training
args = Namespace(
    train_path='./train.csv',
    test_path='./test.csv',
    output_dir="/content/drive/MyDrive/map-charting-student-math-misunderstandings/hunyuan_7b_trained",      
    mode='train',
    model_name="tencent/Hunyuan-7B-Instruct", 
    num_epochs=3,
    learning_rate=2e-4,
    batch_size=16,
    gradient_accumulation_steps=4,
    val_split=0.1
)

DP = DataProcessor(args)
train_df = DP.preprocess()

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=DP.get_num_classes(),
    device_map="auto",
    torch_dtype=torch.bfloat16
)
base_model.config.pad_token_id = tokenizer.pad_token_id

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=64,
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj"]
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

MAX_LEN = 256
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

# Split train/validation
train_data, val_data = train_test_split(train_df[['text', 'label']], test_size=args.val_split, random_state=42)

ds_train = Dataset.from_pandas(train_data)
ds_val = Dataset.from_pandas(val_data)

ds_train = ds_train.map(tokenize_function, batched=True)
ds_val = ds_val.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size * 2,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    bf16=True,
    fp16=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    processing_class=tokenizer,
)

# Train the model
trainer.train()

# Save the final model
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print(f"Training complete! Model saved to {args.output_dir}")