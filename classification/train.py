import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Example data in dictionary format for classification with custom labels
data = {
    "text": ["I love programming.", "This is a sad story."],
    "labels": [1, 0],  # Assuming 1 for positive and 0 for negative
}

# Define custom labels
label_list = ["negative", "positive"]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Convert the DataFrame to a HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_list)
)

# Update the model configuration with the label mappings
model.config.id2label = id2label
model.config.label2id = label2id

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
)

# Train the model
trainer.train()

# Save the final model
model.save_pretrained("./results/final_model")
tokenizer.save_pretrained("./results/final_model")
