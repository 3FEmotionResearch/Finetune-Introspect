# Introspection Dataset Generation and Fine-tuning

This system creates a fine-tuning dataset to give the model self-introspection capabilities, allowing it to predict its own confidence levels for generating specific emotion words.

## Overview

The system transforms your emotion inference results into a dataset where the model learns to answer questions like:
- "If you will be asked this question: [Chinese text]. What is your confidence level that you will answer 'Fear'?"
- Expected answer: "high", "medium", or "low" (based on the percentage from your inference results)

## Confidence Mapping

- **Low confidence**: 0-33.33% (emotions that appeared in few inference runs)
- **Medium confidence**: 33.33-66.67% (emotions that appeared in some inference runs)  
- **High confidence**: 66.67-100% (emotions that appeared in most inference runs)

## Files

### 1. `create_introspection_dataset.py`
Main script that generates the dataset and saves inspection files.

### 2. `run_finetuning.py`
Separate script to run fine-tuning after you've reviewed the dataset.

## Usage

### Step 1: Generate Dataset for Inspection

```bash
python create_introspection_dataset.py
```

This will create a `dataset_inspection/` folder with:
- `raw_dataset.json` - Raw dataset in JSON format
- `formatted_examples.txt` - First 20 examples with chat templates
- `dataset_statistics.txt` - Dataset statistics and distributions
- `confidence_examples.txt` - Examples grouped by confidence level
- `tokenized_samples.txt` - Tokenized samples (what the model sees)

### Step 2: Review the Dataset

Examine the files in `dataset_inspection/` to:
- Verify the input/output format looks correct
- Check the confidence level distribution
- Ensure the tokenization is working properly
- Review example training pairs

### Step 3: Run Fine-tuning

After reviewing the dataset, install the required library and run fine-tuning:

```bash
pip install datasets
python run_finetuning.py
```

## Expected Dataset Structure

Each training example will look like:

**Input**: 
```
If you will be asked this question: 把我给抓回去那我就完了我肯定会受到责罚的. What is your confidence level that you will answer Fear?
```

**Output**: 
```
high
```

## Fine-tuning Parameters

- **Model**: meta-llama/Meta-Llama-3-8B-Instruct
- **Epochs**: 3
- **Batch size**: 2 (for memory efficiency)
- **Learning rate**: 2e-5
- **Training split**: 80% train, 20% eval

## Testing

The system includes automatic testing with sample questions to verify the model's introspection capabilities after fine-tuning.

## Output

The fine-tuned model will be saved to `./introspection_model/` and can be used to:
1. Generate emotion predictions for Chinese text
2. Predict confidence levels for specific emotion words
3. Provide self-awareness about its own prediction capabilities

## Requirements

- transformers
- torch
- datasets (install with `pip install datasets`)
- Access to meta-llama/Meta-Llama-3-8B-Instruct (requires HuggingFace token)

## Notes

- The dataset generation depends on your `inference_results/merged_results/` folder
- Each emotion word from your inference results becomes a separate training example
- The model learns to map its historical performance to confidence levels
- Higher percentages in your inference results = higher confidence predictions 