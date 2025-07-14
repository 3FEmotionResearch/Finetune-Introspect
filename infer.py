import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_PATH = "data/your_dataset.csv"  # Update this with your actual dataset filename


def load_model():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Please add your CSV file.")
        return
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with {len(df)} rows.")
    # TODO: Add inference code here once dataset is provided

if __name__ == "__main__":
    main() 