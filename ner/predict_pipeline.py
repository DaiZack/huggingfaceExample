from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load the fine-tuned model and tokenizer
model_path = "./results/final_model"  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Initialize the NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Use the NER pipeline on new text
text = "John lives in New York and Sarah is from Paris."
ner_results = ner_pipeline(text)

# Print the results
for entity in ner_results:
    print(
        f"Entity: {entity['word']}, Label: {entity['entity']}, Score: {entity['score']:.2f}"
    )
