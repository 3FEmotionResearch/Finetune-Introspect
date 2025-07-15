import json
import os
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from datetime import datetime

# Note: You may need to install the datasets library: pip install datasets
try:
    from datasets import Dataset
    HAS_DATASETS = True
except ImportError:
    print("Warning: datasets library not found. Some functionality may be limited.")
    print("Install with: pip install datasets")
    HAS_DATASETS = False

def load_merged_results(merged_results_dir):
    """
    Load all merged result files from the directory.
    
    Args:
        merged_results_dir (str): Path to merged results directory
        
    Returns:
        list: List of dictionaries containing question data
    """
    results = []
    
    for filename in os.listdir(merged_results_dir):
        if filename.endswith('_merged.json'):
            filepath = os.path.join(merged_results_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    print(f"Loaded {len(results)} merged result files")
    return results

def get_confidence_level(percentage):
    """
    Convert percentage to confidence level.
    
    Args:
        percentage (float): Percentage value
        
    Returns:
        str: Confidence level (low, medium, high)
    """
    if percentage <= 33.3333:
        return "low"
    elif percentage <= 66.6666:
        return "medium"
    else:
        return "high"

def generate_introspection_dataset(merged_results_dir, output_file="introspection_dataset.json"):
    """
    Generate introspection dataset from merged results.
    
    Args:
        merged_results_dir (str): Path to merged results directory
        output_file (str): Output file for the dataset
        
    Returns:
        list: List of training examples
    """
    results = load_merged_results(merged_results_dir)
    dataset = []
    
    total_pairs = 0
    
    for result in results:
        question_id = result['question_id']
        chinese_text = result['chinese_text']
        emotion_details = result['emotion_statistics']['emotion_details']
        
        # Create training pair for each emotion word
        for emotion_word, stats in emotion_details.items():
            percentage = stats['percentage']
            confidence = get_confidence_level(percentage)
            
            # Create the input prompt with improved format
            input_text = f"If you are asked to analyze the emotion of this sentence: {chinese_text}. What is your confidence level that you will answer emotion word \"{emotion_word}\"?"
            
            # Create training example
            training_example = {
                "question_id": question_id,
                "chinese_text": chinese_text,
                "emotion_word": emotion_word,
                "percentage": percentage,
                "confidence": confidence,
                "input": input_text,
                "output": confidence
            }
            
            dataset.append(training_example)
            total_pairs += 1
    
    print(f"Generated {total_pairs} training pairs from {len(results)} questions")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to {output_file}")
    return dataset

def prepare_training_data_from_formatted(formatted_dataset, tokenizer, max_length=512):
    """
    Prepare training data directly from pre-formatted dataset.
    
    Args:
        formatted_dataset (list): List of pre-formatted training examples with "text" field
        tokenizer: Tokenizer for the model
        max_length (int): Maximum sequence length
        
    Returns:
        Dataset: Prepared dataset for training
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library is required for training. Install with: pip install datasets")
    
    # Tokenize the data directly
    def tokenize_function(example):
        # Process single example at a time
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",  # Pad to max_length for consistency
            max_length=max_length,
            return_tensors=None  # Don't return tensors, just lists
        )
        
        # Set labels to be the same as input_ids for causal language modeling
        # Create a proper copy as a new list, not a reference
        tokenized["labels"] = list(tokenized["input_ids"])
        
        return tokenized
    
    # Create dataset directly from formatted examples
    try:
        # Convert list of dicts to dict of lists for datasets 2.4.0
        data_dict = {}
        for key in formatted_dataset[0].keys():
            data_dict[key] = [example[key] for example in formatted_dataset]
        
        hf_dataset = Dataset.from_dict(data_dict)
        tokenized_dataset = hf_dataset.map(tokenize_function, batched=False)  # Process one at a time
        
        # Remove unnecessary columns - only keep what the model needs
        columns_to_remove = [col for col in tokenized_dataset.column_names 
                           if col not in ['input_ids', 'attention_mask', 'labels']]
        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        
        return tokenized_dataset
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("Make sure you have the datasets library installed: pip install datasets")
        raise

def prepare_training_data(dataset, tokenizer, max_length=512):
    """
    Prepare training data for fine-tuning.
    
    Args:
        dataset (list): List of training examples
        tokenizer: Tokenizer for the model
        max_length (int): Maximum sequence length
        
    Returns:
        Dataset: Prepared dataset for training
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library is required for training. Install with: pip install datasets")
    
    def format_example(example):
        """Format example for training"""
        input_text = example['input']
        output_text = example['output']
        
        # Create the simple conversation format without system message
        full_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"
        
        return {"text": full_text}
    
    # Format all examples
    formatted_data = [format_example(example) for example in dataset]
    
    # Tokenize the data
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None  # Don't return tensors, just lists
        )
        
        # Set labels to be the same as input_ids for causal language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Create dataset
    try:
        # Convert list of dicts to dict of lists for datasets 2.4.0
        data_dict = {}
        for key in formatted_data[0].keys():
            data_dict[key] = [example[key] for example in formatted_data]
        
        hf_dataset = Dataset.from_dict(data_dict)
        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("Make sure you have the datasets library installed: pip install datasets")
        raise

def fine_tune_model(dataset_file, model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
                   output_dir="./introspection_model", num_epochs=3, 
                   batch_size=4, learning_rate=5e-5, use_formatted_data=False):
    """
    Fine-tune the model for introspection capabilities.
    
    Args:
        dataset_file (str): Path to the dataset JSON file
        model_name (str): Name of the base model
        output_dir (str): Directory to save fine-tuned model
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate
        use_formatted_data (bool): Whether to use pre-formatted data (formatted_for_training.json)
    """
    print(f"Starting fine-tuning with model: {model_name}")
    
    # Load dataset
    if use_formatted_data:
        # Use pre-formatted data that's ready for training
        formatted_file = dataset_file.replace("introspection_dataset.json", "formatted_for_training.json")
        if os.path.exists(formatted_file):
            print(f"Using pre-formatted data: {formatted_file}")
            with open(formatted_file, 'r', encoding='utf-8') as f:
                formatted_dataset = json.load(f)
            
            # Convert to the format expected by prepare_training_data
            dataset = []
            for item in formatted_dataset:
                dataset.append({
                    "question_id": item["question_id"],
                    "chinese_text": item["chinese_text"],
                    "emotion_word": item["emotion_word"],
                    "percentage": item["percentage"],
                    "confidence": item["confidence"],
                    "input": item["text"].split("<|start_header_id|>user<|end_header_id|>")[1].split("<|eot_id|>")[0].strip(),
                    "output": item["confidence"]
                })
        else:
            print(f"Pre-formatted file not found: {formatted_file}")
            print("Using raw dataset instead...")
            with open(dataset_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
    else:
        # Use raw dataset
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} training examples")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=True
    )
    
    # Prepare training data
    tokenized_dataset = prepare_training_data(dataset, tokenizer)
    
    # Split dataset (80% train, 20% eval)
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Evaluation set size: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb logging
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True,  # Enable mixed precision training
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Fine-tuning completed! Model saved to {output_dir}")

def fine_tune_model_direct(formatted_data_file, model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
                          output_dir="./introspection_model", num_epochs=3, 
                          batch_size=4, learning_rate=5e-5, max_train_samples=None):
    """
    Fine-tune the model directly using pre-formatted data.
    
    Args:
        formatted_data_file (str): Path to the formatted_for_training.json file
        model_name (str): Name of the base model
        output_dir (str): Directory to save fine-tuned model
        num_epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate
        max_train_samples (int, optional): Limit dataset to N samples for faster testing
    """
    print(f"Starting fine-tuning with model: {model_name}")
    print(f"Using pre-formatted data: {formatted_data_file}")
    
    # Load pre-formatted dataset
    with open(formatted_data_file, 'r', encoding='utf-8') as f:
        formatted_dataset = json.load(f)
    
    # Limit dataset size if specified (for faster testing)
    if max_train_samples is not None:
        formatted_dataset = formatted_dataset[:max_train_samples]
        print(f"Limited dataset to {max_train_samples} samples for faster testing")
    
    print(f"Loaded {len(formatted_dataset)} pre-formatted training examples")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use GPU - CUDA is now working properly with PyTorch 2.0.1+cu117
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use GPU-optimized model settings for A100
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 for GPU efficiency
        token=True
    )
    
    # Prepare training data directly from formatted data
    tokenized_dataset = prepare_training_data_from_formatted(formatted_dataset, tokenizer, max_length=512)
    
    # Split dataset (80% train, 20% eval)
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Evaluation set size: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=10,  # Reduced warmup steps for faster start
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=1,  # Log after every step to see progress
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=10,  # Evaluate every 10 steps (reduced from 100)
        save_strategy="steps",
        save_steps=50,  # Save more frequently (reduced from 500)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Disable wandb logging
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        fp16=True,  # Enable mixed precision for GPU training
        gradient_checkpointing=False,  # Disable for better speed on A100
        dataloader_num_workers=4,  # Use more workers for better data loading
    )
    
    # Create trainer (using default data collator)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Fine-tuning completed! Model saved to {output_dir}")

def test_introspection_model(model_dir, test_questions):
    """
    Test the fine-tuned introspection model.
    
    Args:
        model_dir (str): Directory containing the fine-tuned model
        test_questions (list): List of test questions
    """
    print(f"Loading fine-tuned model from {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    for question in test_questions:
        chinese_text = question["chinese_text"]
        emotion_word = question["emotion_word"]
        
        # Use the improved prompt format without system message
        user_prompt = f"If you are asked to analyze the emotion of this sentence: {chinese_text}. What is your confidence level that you will answer emotion word \"{emotion_word}\"?"
        
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
      )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("assistant<|end_header_id|>")[-1].strip()
        
        print(f"Sentence: {chinese_text}")
        print(f"Emotion: {emotion_word}")
        print(f"Model confidence: {assistant_response}")
        print("-" * 50)

def save_formatted_dataset(dataset, output_dir="dataset_inspection"):
    """
    Save the formatted dataset in various formats for inspection.
    
    Args:
        dataset (list): List of training examples
        output_dir (str): Directory to save inspection files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw dataset as JSON (already done in generate_introspection_dataset)
    raw_file = os.path.join(output_dir, "raw_dataset.json")
    with open(raw_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # Save ALL formatted examples (with chat templates) as text - CHANGED: Save all, not just first 20
    formatted_file = os.path.join(output_dir, "formatted_examples.txt")
    with open(formatted_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ALL FORMATTED DATASET EXAMPLES FOR FINE-TUNING\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total examples: {len(dataset)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, example in enumerate(dataset):  # CHANGED: Save ALL examples, not just first 20
            input_text = example['input']
            output_text = example['output']
            
            # Create the simple conversation format without system message
            full_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"
            
            f.write(f"EXAMPLE {i+1}:\n")
            f.write(f"Question ID: {example['question_id']}\n")
            f.write(f"Chinese Text: {example['chinese_text']}\n")
            f.write(f"Emotion Word: {example['emotion_word']}\n")
            f.write(f"Percentage: {example['percentage']:.1f}%\n")
            f.write(f"Confidence Level: {example['confidence']}\n")
            f.write(f"Full Training Text:\n")
            f.write(full_text)
            f.write("\n" + "=" * 80 + "\n\n")
    
    # Save formatted examples as JSON (ready for direct use in fine-tuning)
    formatted_json_file = os.path.join(output_dir, "formatted_for_training.json")
    formatted_examples = []
    for example in dataset:
        input_text = example['input']
        output_text = example['output']
        
        # Create the simple conversation format without system message
        full_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"
        
        formatted_examples.append({
            "text": full_text,
            "question_id": example['question_id'],
            "chinese_text": example['chinese_text'],
            "emotion_word": example['emotion_word'],
            "percentage": example['percentage'],
            "confidence": example['confidence']
        })
    
    with open(formatted_json_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_examples, f, ensure_ascii=False, indent=2)
    
    # Save as CSV for direct fine-tuning use
    csv_file = os.path.join(output_dir, "formatted_for_training.csv")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['input_text', 'output_text', 'question_id', 'chinese_text', 'emotion_word', 'percentage', 'confidence'])
        
        # Write data rows
        for example in dataset:
            input_text = example['input']
            output_text = example['output']
            
            writer.writerow([
                input_text,
                output_text,
                example['question_id'],
                example['chinese_text'],
                example['emotion_word'],
                example['percentage'],
                example['confidence']
            ])
    
    # Save as simple CSV (just input/output for basic fine-tuning)
    simple_csv_file = os.path.join(output_dir, "simple_training_data.csv")
    with open(simple_csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['input', 'output'])
        
        # Write data rows
        for example in dataset:
            writer.writerow([
                example['input'],
                example['output']
            ])
    
    # Save dataset statistics
    stats_file = os.path.join(output_dir, "dataset_statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("DATASET STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        
        # Total counts
        f.write(f"Total training pairs: {len(dataset)}\n")
        f.write(f"Unique questions: {len(set(ex['question_id'] for ex in dataset))}\n")
        f.write(f"Unique emotions: {len(set(ex['emotion_word'] for ex in dataset))}\n\n")
        
        # Confidence distribution
        confidence_counts = {}
        for ex in dataset:
            conf = ex['confidence']
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        f.write("Confidence Level Distribution:\n")
        for conf, count in sorted(confidence_counts.items()):
            percentage = (count / len(dataset)) * 100
            f.write(f"  {conf}: {count} ({percentage:.1f}%)\n")
        
        f.write("\n")
        
        # Top emotions
        emotion_counts = {}
        for ex in dataset:
            emotion = ex['emotion_word']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        f.write("Top 20 Most Frequent Emotions:\n")
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
        for emotion, count in sorted_emotions[:20]:
            f.write(f"  {emotion}: {count} instances\n")
        
        f.write("\n")
        
        # Percentage distribution
        f.write("Percentage Distribution:\n")
        percentages = [ex['percentage'] for ex in dataset]
        f.write(f"  Min: {min(percentages):.1f}%\n")
        f.write(f"  Max: {max(percentages):.1f}%\n")
        f.write(f"  Mean: {sum(percentages)/len(percentages):.1f}%\n")
        f.write(f"  Median: {sorted(percentages)[len(percentages)//2]:.1f}%\n")
    
    # Save confidence level examples
    confidence_examples_file = os.path.join(output_dir, "confidence_examples.txt")
    with open(confidence_examples_file, 'w', encoding='utf-8') as f:
        f.write("CONFIDENCE LEVEL EXAMPLES\n")
        f.write("=" * 50 + "\n\n")
        
        for conf_level in ['low', 'medium', 'high']:
            f.write(f"{conf_level.upper()} CONFIDENCE EXAMPLES:\n")
            f.write("-" * 30 + "\n")
            
            examples = [ex for ex in dataset if ex['confidence'] == conf_level][:10]  # Show more examples
            for i, ex in enumerate(examples):
                f.write(f"{i+1}. {ex['chinese_text']}\n")
                f.write(f"   Emotion: {ex['emotion_word']} ({ex['percentage']:.1f}%)\n")
                f.write(f"   Input: {ex['input']}\n")
                f.write(f"   Output: {ex['output']}\n\n")
            
            f.write("\n")
    
    print(f"Dataset inspection files saved to {output_dir}/")
    print(f"  - raw_dataset.json: Raw dataset in JSON format")
    print(f"  - formatted_examples.txt: ALL {len(dataset)} examples with chat templates")
    print(f"  - formatted_for_training.json: Ready-to-use format for fine-tuning")
    print(f"  - formatted_for_training.csv: CSV format with all metadata")
    print(f"  - simple_training_data.csv: Simple CSV (input/output only)")
    print(f"  - dataset_statistics.txt: Dataset statistics and distributions")
    print(f"  - confidence_examples.txt: Examples grouped by confidence level")

def save_tokenized_samples(dataset, tokenizer, output_dir="dataset_inspection", num_samples=5):
    """
    Save tokenized samples to show what the model will actually see.
    
    Args:
        dataset (list): List of training examples
        tokenizer: Tokenizer for the model
        output_dir (str): Directory to save inspection files
        num_samples (int): Number of samples to save
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tokenized_file = os.path.join(output_dir, "tokenized_samples.txt")
    with open(tokenized_file, 'w', encoding='utf-8') as f:
        f.write("TOKENIZED SAMPLES\n")
        f.write("=" * 50 + "\n\n")
        f.write("This shows what the model will actually see during training.\n\n")
        
        for i, example in enumerate(dataset[:num_samples]):
            input_text = example['input']
            output_text = example['output']
            
            # Create the full conversation format
            full_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output_text}<|eot_id|>"
            
            # Tokenize
            tokenized = tokenizer(full_text, truncation=True, padding=False)
            
            f.write(f"SAMPLE {i+1}:\n")
            f.write(f"Question ID: {example['question_id']}\n")
            f.write(f"Emotion: {example['emotion_word']} ({example['percentage']:.1f}%) -> {example['confidence']}\n")
            f.write(f"Text length: {len(full_text)} characters\n")
            f.write(f"Token count: {len(tokenized['input_ids'])} tokens\n\n")
            
            f.write("Original text:\n")
            f.write(full_text)
            f.write("\n\n")
            
            f.write("Token IDs:\n")
            f.write(str(tokenized['input_ids']))
            f.write("\n\n")
            
            f.write("Decoded tokens:\n")
            for token_id in tokenized['input_ids']:
                token = tokenizer.decode([token_id])
                f.write(f"[{token_id}] '{token}'\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"Tokenized samples saved to {tokenized_file}")

def generate_and_save_dataset(merged_results_dir, output_dir="dataset_inspection"):
    """
    Generate dataset and save all inspection files.
    
    Args:
        merged_results_dir (str): Path to merged results directory
        output_dir (str): Directory to save inspection files
    """
    print("Generating introspection dataset...")
    
    # Generate dataset
    dataset = generate_introspection_dataset(merged_results_dir, 
                                           os.path.join(output_dir, "introspection_dataset.json"))
    
    # Save formatted dataset for inspection
    save_formatted_dataset(dataset, output_dir)
    
    # Load tokenizer and save tokenized samples
    print("Loading tokenizer for tokenized samples...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        save_tokenized_samples(dataset, tokenizer, output_dir, num_samples=10)
    except Exception as e:
        print(f"Could not load tokenizer for tokenized samples: {e}")
        print("You can still review the formatted examples and statistics.")
    
    print(f"\nDataset generation complete! Review the files in {output_dir}/ before fine-tuning.")
    return dataset

if __name__ == "__main__":
    # Configuration
    MERGED_RESULTS_DIR = "inference_results/merged_results"
    DATASET_INSPECTION_DIR = "dataset_inspection"
    MODEL_OUTPUT_DIR = "./introspection_model"
    
    # Step 1: Generate and save dataset for inspection
    print("Step 1: Generating introspection dataset and saving inspection files...")
    dataset = generate_and_save_dataset(MERGED_RESULTS_DIR, DATASET_INSPECTION_DIR)
    
    print(f"\n{'='*60}")
    print("DATASET GENERATED FOR INSPECTION!")
    print(f"{'='*60}")
    print(f"Please review the files in {DATASET_INSPECTION_DIR}/ before proceeding:")
    print(f"  📄 raw_dataset.json - Raw dataset in JSON format")
    print(f"  📝 formatted_examples.txt - ALL {len(dataset)} examples with chat templates")
    print(f"  🚀 formatted_for_training.json - Ready-to-use format for fine-tuning")
    print(f"  📊 formatted_for_training.csv - CSV format with all metadata")
    print(f"  📋 simple_training_data.csv - Simple CSV (input/output only)")
    print(f"  📈 dataset_statistics.txt - Dataset statistics and distributions")
    print(f"  🔍 confidence_examples.txt - Examples grouped by confidence level")
    print(f"  🔤 tokenized_samples.txt - Tokenized samples (what model sees)")
    print(f"\nTo proceed with fine-tuning, install datasets library first:")
    print(f"  pip install datasets")
    print(f"  python run_finetuning.py")
    print(f"{'='*60}")
    
    # Step 2: Fine-tune the model (commented out for inspection)
    # Uncomment the following lines after reviewing the dataset
    """
    print("\nStep 2: Fine-tuning the model...")
    fine_tune_model(
        dataset_file=os.path.join(DATASET_INSPECTION_DIR, "introspection_dataset.json"),
        output_dir=MODEL_OUTPUT_DIR,
        num_epochs=3,
        batch_size=2,  # Smaller batch size for memory efficiency
        learning_rate=2e-5
    )
    
    # Step 3: Test the model
    print("\nStep 3: Testing the introspection model...")
    test_questions = [
        {
            "chinese_text": "把我给抓回去那我就完了我肯定会受到责罚的",
            "emotion_word": "Fear"
        },
        {
            "chinese_text": "把我给抓回去那我就完了我肯定会受到责罚的", 
            "emotion_word": "Anxiety"
        },
        {
            "chinese_text": "把我给抓回去那我就完了我肯定会受到责罚的",
            "emotion_word": "Joy"
        }
    ]
    
    test_introspection_model(MODEL_OUTPUT_DIR, test_questions)
    """ 