import csv
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import logging

# Global configuration - easily changeable
NUM_INFERENCE_RUNS = 10  # Change this number to run inference N times per question
MAX_NEW_TOKENS = 30
# Model options (uncomment the one you want to use):
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # Requires HF login and access request
# MODEL_NAME = "microsoft/DialoGPT-medium"  # Public, no auth needed
# MODEL_NAME = "google/flan-t5-base"  # Public, good for instruction following
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Public, good alternative
# MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"  # Public, good for conversations

# Global model and tokenizer (loaded once)
model = None
tokenizer = None

def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup individual run logger
    run_logger = logging.getLogger('run_logger')
    run_logger.setLevel(logging.INFO)
    run_handler = logging.FileHandler(os.path.join(output_dir, 'individual_runs.log'))
    run_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    run_logger.addHandler(run_handler)
    
    # Setup aggregate logger
    agg_logger = logging.getLogger('aggregate_logger')
    agg_logger.setLevel(logging.INFO)
    agg_handler = logging.FileHandler(os.path.join(output_dir, 'aggregate_progress.log'))
    agg_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    agg_logger.addHandler(agg_handler)
    
    return run_logger, agg_logger

def load_model():
    """Load the model and tokenizer"""
    global model, tokenizer
    if model is None:
        print(f"Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=True)
        
        # Use direct import for Meta-Llama models
        if "meta-llama" in MODEL_NAME.lower():
            from transformers.models.llama.modeling_llama import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                token=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                token=True
            )
        print("Model loaded successfully!")

def infer_emotion_labels(text):
    """
    Infer emotion labels from Chinese text using the specified model.
    
    Args:
        text (str): Chinese text to analyze
        
    Returns:
        tuple: (emotion_labels_list, raw_model_output)
    """
    # Load model if not already loaded
    if model is None or tokenizer is None:
        load_model()
    
    # Ensure model and tokenizer are loaded
    if model is None or tokenizer is None:
        raise RuntimeError("Failed to load model or tokenizer")
    
    # Clear prompt for emotion analysis
    prompt = f"""Analyze the emotions in this Chinese text. Respond with English emotion words only.

Text: {text}

Emotions: """
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the model's response (after the prompt)
    if "Emotions: " in full_response:
        raw_model_output = full_response.split("Emotions: ")[-1].strip()
    else:
        raw_model_output = full_response.strip()
    
    # Basic parsing - just split by common separators
    if raw_model_output:
        # Split by comma, semicolon, and "and"
        separators = [",", ";", " and "]
        emotion_words = [raw_model_output]
        
        for separator in separators:
            new_words = []
            for word in emotion_words:
                new_words.extend(word.split(separator))
            emotion_words = new_words
        
        # Clean up each word (remove extra spaces)
        emotions = []
        for word in emotion_words:
            clean_word = word.strip()
            if clean_word:
                emotions.append(clean_word)
        
        return emotions[:5], raw_model_output  # Limit to 5 words
    
    return ["Unknown"], raw_model_output

def run_inference(csv_path, output_dir, num_runs=NUM_INFERENCE_RUNS, model_name=MODEL_NAME):
    """
    Run emotion inference on Chinese text from CSV file.
    
    Args:
        csv_path (str): Path to CSV file containing Chinese text
        output_dir (str): Directory to save inference results
        num_runs (int): Number of times to run inference for each question
        model_name (str): Name/path of the model to use for inference
    """
    global MODEL_NAME
    MODEL_NAME = model_name
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    raw_output_dir = os.path.join(output_dir, "raw_outputs")
    merged_output_dir = os.path.join(output_dir, "merged_results")
    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(merged_output_dir, exist_ok=True)
    
    # Setup logging
    run_logger, agg_logger = setup_logging(output_dir)
    
    # Load model before processing
    load_model()
    
    # Log experiment start
    agg_logger.info(f"Starting experiment with model: {MODEL_NAME}")
    agg_logger.info(f"Number of runs per question: {num_runs}")
    
    # Overall summary data
    overall_summary = {
        "experiment_info": {
            "model_used": MODEL_NAME,
            "num_runs": num_runs,
            "timestamp": datetime.now().isoformat(),
            "total_questions": 0
        },
        "results": []
    }
    
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        total_questions = len(list(reader))
        csvfile.seek(0)
        next(reader)  # Skip header again
        
        agg_logger.info(f"Total questions to process: {total_questions}")
        
        for question_idx, row in enumerate(reader, 1):
            question_id = row['name']
            chinese_text = row['chinese']
            
            print(f"Processing {question_id}: {chinese_text[:50]}...")
            agg_logger.info(f"Processing question {question_idx}/{total_questions}: {question_id}")
            
            # Create question-specific directory for raw outputs
            question_raw_dir = os.path.join(raw_output_dir, question_id)
            os.makedirs(question_raw_dir, exist_ok=True)
            
            # Run inference multiple times to collect emotion labels
            results = []
            
            for run_num in range(num_runs):
                print(f"  Run {run_num + 1}/{num_runs}")
                start_time = datetime.now()
                
                labels, raw_output = infer_emotion_labels(chinese_text)
                results.append(labels)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Save individual run output
                run_data = {
                    "run_number": run_num + 1,
                    "question_id": question_id,
                    "chinese_text": chinese_text,
                    "parsed_emotions": labels,
                    "raw_model_output": raw_output,
                    "timestamp": end_time.isoformat(),
                    "duration_seconds": duration
                }
                
                # IMMEDIATE LOGGING - write to log file after each run
                run_logger.info(f"Q:{question_id} | Run:{run_num + 1}/{num_runs} | Emotions:{labels} | Duration:{duration:.2f}s")
                
                # Save individual run to separate file immediately
                run_file = os.path.join(question_raw_dir, f"run_{run_num + 1:02d}.json")
                with open(run_file, "w", encoding="utf-8") as f:
                    json.dump(run_data, f, ensure_ascii=False, indent=2)
            
            # Calculate emotion frequency for confidence analysis
            emotion_counts = {}
            total_emotions = 0
            for run_emotions in results:
                for emotion in run_emotions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    total_emotions += 1
            
            # Calculate frequencies
            emotion_stats = {}
            for emotion, count in emotion_counts.items():
                emotion_stats[emotion] = {
                    "count": count,
                    "frequency": round(count / num_runs, 3),  # Frequency out of total runs
                    "percentage": round(count / num_runs * 100, 1)  # Percentage of runs
                }
            
            # Sort by count (descending)
            sorted_emotions = sorted(emotion_stats.items(), key=lambda x: x[1]['count'], reverse=True)
            
            # Create simplified merged result for this question
            merged_data = {
                "question_id": question_id,
                "chinese_text": chinese_text,
                "emotion_statistics": {
                    "total_runs": num_runs,
                    "total_emotion_instances": total_emotions,
                    "unique_emotions": len(emotion_counts),
                    "emotion_details": dict(sorted_emotions)
                },
                "summary": {
                    "most_frequent_emotion": sorted_emotions[0] if sorted_emotions else None,
                    "top_3_emotions": [emotion for emotion, _ in sorted_emotions[:3]]
                },
                "model_used": MODEL_NAME,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save merged result for this question
            merged_file = os.path.join(merged_output_dir, f"{question_id}_merged.json")
            with open(merged_file, "w", encoding="utf-8") as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
            # Log question completion
            top_emotions = [(emotion, stats['count']) for emotion, stats in sorted_emotions[:3]]
            agg_logger.info(f"Completed question {question_idx}/{total_questions}: {question_id} | Top emotions: {top_emotions}")
            
            # Add to overall summary
            overall_summary["results"].append({
                "question_id": question_id,
                "chinese_text": chinese_text,
                "top_emotions": top_emotions,
                "total_unique_emotions": len(emotion_counts),
                "most_frequent_emotion": sorted_emotions[0] if sorted_emotions else None
            })
            
            overall_summary["experiment_info"]["total_questions"] += 1
    
    # Save overall summary
    summary_file = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)
    
    # Final logging
    agg_logger.info("Experiment completed successfully!")
    agg_logger.info(f"Total questions processed: {overall_summary['experiment_info']['total_questions']}")
    
    print(f"\nInference completed successfully!")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"📄 Raw outputs: {raw_output_dir}/")
    print(f"📊 Merged results: {merged_output_dir}/")
    print(f"📋 Overall summary: {summary_file}")
    print(f"📝 Individual runs log: {os.path.join(output_dir, 'individual_runs.log')}")
    print(f"📈 Aggregate progress log: {os.path.join(output_dir, 'aggregate_progress.log')}")
    print(f"🔢 Total questions processed: {overall_summary['experiment_info']['total_questions']}")
    print(f"🤖 Each question inferred {num_runs} times using model: {MODEL_NAME}")

if __name__ == "__main__":
    run_inference(
        csv_path="../data/subtitles.csv",  # Fixed path to go up one directory
        output_dir="inference_results",
        num_runs=NUM_INFERENCE_RUNS,
        model_name=MODEL_NAME
    ) 