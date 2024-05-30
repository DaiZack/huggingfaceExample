from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the fine-tuned model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained("./results/final_model")

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def ner_text(text):
    # Tokenize the input text
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Move tensors to the same device as the model
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**tokens)

    # Get the predicted labels
    predictions = torch.argmax(outputs.logits, dim=2)

    # Convert predictions to labels
    predicted_labels = [
        model.config.id2label[label_id.item()] for label_id in predictions[0]
    ]

    # Get the input tokens
    input_tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

    return list(zip(input_tokens, predicted_labels))


# Use the NER function on new text
text = "John lives in New York and Sarah is from Paris."
ner_results = ner_text(text)

# Print the results
for token, label in ner_results:
    print(f"{token}: {label}")
