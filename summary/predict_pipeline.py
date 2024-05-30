from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model and tokenizer
model_path = "./results/final_model"  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Initialize the summarization pipeline
summarization_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)

# Use the summarization pipeline on new text
text = "The quick brown fox jumps over the lazy dog. The lazy dog lies in the sun. Both are happy."
summary_results = summarization_pipeline(text)

# Print the results
print(f"Original Text: {text}")
for summary in summary_results:
    print(f"Summary: {summary['summary_text']}")
