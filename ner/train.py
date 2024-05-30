import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Example data in dictionary format
data = {
    "text": ["John lives in New York.", "Sarah is from Paris."],
    "ner_tags": [
        ["B-PER", "O", "O", "B-LOC", "I-LOC", "O"],
        ["B-PER", "O", "O", "B-LOC", "O"],
    ],
}

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Convert the DataFrame to a HuggingFace Dataset
dataset = Dataset.from_pandas(df)


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

label_list = ["O", "B-PER", "B-LOC", "I-LOC"]
label2id = {label: i for i, label in enumerate(label_list)}


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"], truncation=True, padding="max_length"
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)


model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=4
)

# Define the label mappings
label_list = ["O", "B-PER", "B-LOC", "I-LOC"]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

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

# Step 5: Train the Model
trainer.train()
# Save the final model
model.save_pretrained("./results/final_model")
tokenizer.save_pretrained("./results/final_model")
