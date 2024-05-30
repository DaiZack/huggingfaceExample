import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# Example data in dictionary format for summarization
data = {
    "text": [
        "The quick brown fox jumps over the lazy dog. The lazy dog lies in the sun. Both are happy.",
        "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do.",
    ],
    "summary": [
        "A quick fox jumps over a lazy dog. Both are happy.",
        "Alice was tired of sitting by her sister with nothing to do.",
    ],
}

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Convert the DataFrame to a HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)


# Tokenize the dataset
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["text"], max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        examples["summary"], max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Define the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",  # Log directory
    logging_steps=10,  # Log every 10 steps
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=3,  # Limit the total amount of checkpoints
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the final model
model.save_pretrained("./results/final_model")
tokenizer.save_pretrained("./results/final_model")
