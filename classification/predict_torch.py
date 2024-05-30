from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_name = "./results/final_model"  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify_text(text):
    # Tokenize the input text
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Move tensors to the same device as the model
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**tokens)

    # Get the predicted label
    predictions = torch.argmax(outputs.logits, dim=1)

    # Convert prediction to label
    predicted_label = model.config.id2label[predictions.item()]

    return predicted_label

# Use the classification function on new text
text = "I am happy with this product."
classification_result = classify_text(text)

# Print the result
print(f"Text: {text}")
print(f"Predicted Label: {classification_result}")
