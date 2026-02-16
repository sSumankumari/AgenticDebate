import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. Define your custom local path
custom_path = r"models\vlt5_keywords"

# 2. Load/Download model and tokenizer
# If the path exists and contains the model, it loads locally.
# If not, it downloads to this path.
model = T5ForConditionalGeneration.from_pretrained(
    "Voicelab/vlt5-base-keywords", 
    cache_dir=custom_path
)
tokenizer = T5Tokenizer.from_pretrained(
    "Voicelab/vlt5-base-keywords", 
    cache_dir=custom_path,
    use_fast=False
)

# Optional: To be 100% sure it never checks the internet after the first download,
# you can use local_files_only=True after the first run:
# model = T5ForConditionalGeneration.from_pretrained(custom_path, local_files_only=True)

task_prefix = "Keywords: "
inputs = [
    "The transition to a Green Economy requires a massive shift in global subsidies from fossil fuels to renewable energy infrastructure. However, this transition risks 'Greenflation'—a period of high inflation caused by the rising costs of essential metals like lithium and copper needed for batteries. Policymakers must balance the urgent need for Decarbonization against the immediate risk of Social Unrest due to rising energy prices.",
]

for sample in inputs:
    input_sequences = [task_prefix + sample]
    input_ids = tokenizer(
        input_sequences, return_tensors="pt", truncation=True
    ).input_ids
    
    output = model.generate(
        input_ids, 
        no_repeat_ngram_size=3, 
        num_beams=3,
        max_length=100,
        temperature=0.7,   # Add this
        do_sample=True,
    )
    
    predicted = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nInput: {sample[:50]}...")
    print(f"---> Predicted Keywords: {predicted}")