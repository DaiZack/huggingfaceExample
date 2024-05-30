from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Load the fine-tuned model and tokenizer
model_path = "./results/final_model"  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Initialize the classification pipeline
classification_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Use the classification pipeline on new text
text = "I am happy with this product."
classification_results = classification_pipeline(text)

# Print the results
for result in classification_results:
    print(
        f"Label: {result['label']}, Score: {result['score']:.2f}"
    )
