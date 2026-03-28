"""
evaluate_model.py
══════════════════════════════════════════════════════════════════════════════
A simple evaluation script to measure Perplexity (PPL) on a validation dataset.
This demonstrates the 'Performance Measurement' component of the assignment.
══════════════════════════════════════════════════════════════════════════════
"""

import torch
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def evaluate_perplexity(model_id, step_subfolder, dataset_name, split="test", max_samples=100):
    print(f"Loading tokenizer and model from {model_id} ({step_subfolder})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=step_subfolder)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        subfolder=step_subfolder,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    print(f"Loading dataset {dataset_name}...")
    # Using Wikipedia Thai for evaluation context
    ds = load_dataset(dataset_name, "20231101.th", split=split, streaming=True)
    
    nlls = []
    count = 0
    
    print("Calculating Perplexity...")
    for example in tqdm(ds.take(max_samples)):
        text = example["text"]
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
        
        count += 1

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"\nResults for {step_subfolder}:")
    print(f"  - Average Loss: {torch.stack(nlls).mean():.4f}")
    print(f"  - Perplexity:   {ppl.item():.4f}")
    return ppl.item()

if __name__ == "__main__":
    REPO_ID = "Phonsiri/typhoon-3.5b-cpt-ckpt"
    # Update with your desired checkpoint step
    LATEST_STEP = "step_0000275" 
    
    evaluate_perplexity(
        REPO_ID, 
        LATEST_STEP, 
        "wikimedia/wikipedia", 
        max_samples=50
    )
