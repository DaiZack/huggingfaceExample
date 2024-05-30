import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model and tokenizer
model_path = "./results/final_model"  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate summary
def summarize_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)

    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, length_penalty=2.0, early_stopping=True)

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Use the summarization function on new text
text = "The quick brown fox jumps over the lazy dog. The lazy dog lies in the sun. Both are happy."
summary_result = summarize_text(text)

# Print the results
print(f"Original Text: {text}")
print(f"Summary: {summary_result}")
