import argparse
import sys
import os
import time
import json
from datetime import datetime

from scripts.combined_extractor import CombinedExtractor
from scripts.numeric_extractor import NumericExtractor
from scripts.context_extractor import ContextExtractor
from scripts.data_processor import DataProcessor
from scripts.evaluator import Evaluator
from scripts.utils import *

def get_core_model_name(path: str, prefix_to_strip: str) -> str:
    """Extracts the core name from a model path by stripping known prefixes."""
    if not path:
        return "unknown"
    base_name = os.path.basename(path.rstrip('/'))
    if base_name.startswith(prefix_to_strip):
        return base_name[len(prefix_to_strip):]
    return base_name

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='ML model operations script')
    
    # Add the operation argument
    parser.add_argument('--operation', required=True, choices=['train', 'test', 'predict'], 
                        help='Operation to perform')
    
    # Add the type argument (Default: both)
    parser.add_argument('--type', choices=['numeric', 'context', 'both'], default='both',
                        help='Type of training (Default: both)')
    
    # Add config path argument
    parser.add_argument('--config_path', default='config/config_roberta-base.cfg', 
                        help='Path to the spaCy training config file (e.g., config/config_roberta-base.cfg)')
    
    # Argument for Timestamp Control
    parser.add_argument('--use_timestamp', action='store_true', 
                        help='Use a timestamp suffix in the default output/model paths (Default: False)')

    # Model Path Arguments (Used for train and predict)
    parser.add_argument('--model_path', help='Path to the model (for single model operations)')
    parser.add_argument('--numeric_model_path', help='Path to the numeric model')
    parser.add_argument('--context_model_dir', help='Directory path to the context model')
    
    # Data File Arguments
    parser.add_argument('--file_path', help='Path to the input data file (Required for train/predict)')
    parser.add_argument('--pred_path', help='Path to the prediction file (Required for test operation)')
    parser.add_argument('--gold_path', help='Path to the gold/correct answer file (Required for test operation)')

    # Output Path Argument
    parser.add_argument('--output_path', help='Path to save prediction/evaluation results')
    
    # GPU selection
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use (spaCy --gpu-id)')
    
    # Evaluation Arguments
    parser.add_argument('--numeric_labels', default='labels/numeric_labels.txt', 
                        help='Path to numeric labels file (for test operation)')
    parser.add_argument('--match_type', choices=['exact', 'boundary'], default='exact',
                        help='Span matching type for evaluation (for test operation)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get suffix based on argument
    suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if args.use_timestamp else ""
    
    # Extract base model name from config
    default_config_name = os.path.basename(args.config_path).replace('config_', '').replace('.cfg', '')

    # --- Validate Arguments per Operation ---
    if args.operation == 'train':
        if not args.file_path:
            print("Error: Train operation requires --file_path")
            sys.exit(1)
            
        # Dynamic Default Assignment for Training
        if args.type == 'both':
            if not args.numeric_model_path:
                args.numeric_model_path = f"model/numeric_model_{default_config_name}{suffix}"
                print(f"Using default numeric model path: {args.numeric_model_path}")
            if not args.context_model_dir:
                args.context_model_dir = f"model/context_model_{default_config_name}{suffix}"
                print(f"Using default context model directory: {args.context_model_dir}")
        elif args.type in ['numeric', 'context'] and not args.model_path:
            args.model_path = f"model/{args.type}_model_{default_config_name}{suffix}"
            print(f"Using default single model path: {args.model_path}")

    elif args.operation == 'test':
        if not args.pred_path:
            print("Error: Test operation requires --pred_path (prediction file)")
            sys.exit(1)
        if not args.gold_path:
            print("Error: Test operation requires --gold_path (gold/correct answer file)")
            sys.exit(1)

    elif args.operation == 'predict':
        if not args.file_path:
            print("Error: Predict operation requires --file_path (input file)")
            sys.exit(1)
        
        # Determine if using combined models
        using_combined_models = False
        if args.numeric_model_path and args.context_model_dir:
            using_combined_models = True
        elif not args.model_path and not (args.numeric_model_path and args.context_model_dir):
            print("Warning: No model path provided for predict operation. This will likely fail.")

    # --- Execute Operation ---
    print(f"Operation: {args.operation}")

    if args.operation == 'train':
        print(f"Training type: {args.type}")
        if args.type == 'both':
            # Create directories
            os.makedirs(args.context_model_dir, exist_ok=True)
            train_both(args.numeric_model_path, args.context_model_dir, args.file_path, args.config_path, gpu_id=args.gpu_id)
        else:
            if args.type == 'numeric':
                train_numeric(args.model_path, args.file_path, args.config_path, gpu_id=args.gpu_id)
            elif args.type == 'context':
                train_context(args.model_path, args.file_path, args.config_path, gpu_id=args.gpu_id)
    
    elif args.operation == 'test':
        # --- Test (Evaluation) Logic ---
        
        # Generate default output filename based on prediction filename
        pred_basename = os.path.splitext(os.path.basename(args.pred_path))[0]
        
        # Use suffix only if --use_timestamp is set
        output_suffix = suffix if args.use_timestamp else ""
        default_output_path = f"evaluations/evaluation_{pred_basename}_{args.match_type}{output_suffix}.json"
        
        output_path = args.output_path if args.output_path else default_output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Evaluation results path: {output_path}")
        
        evaluate_files(
            pred_path=args.pred_path,
            gold_path=args.gold_path,
            numeric_labels_path=args.numeric_labels,
            match_type=args.match_type,
            output_path=output_path,
            use_timestamp=args.use_timestamp
        )
    
    elif args.operation == 'predict':
        
        # --- Generate Concise Model Identifier ---
        if using_combined_models:
            num_core = get_core_model_name(args.numeric_model_path, "numeric_model_")
            ctx_core = get_core_model_name(args.context_model_dir, "context_model_")
            
            if num_core == ctx_core:
                model_identifier = num_core
            else:
                model_identifier = f"num-{num_core}_ctx-{ctx_core}"
        else:
            model_identifier = os.path.basename(args.model_path) if args.model_path else "single"
        
        # Use suffix only if --use_timestamp is set
        output_suffix = suffix if args.use_timestamp else ""
        default_output_path = f"predictions/predictions_{model_identifier}{output_suffix}.jsonl"
        
        output_path = args.output_path if args.output_path else default_output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Prediction output path: {output_path}")
            
        if using_combined_models:
            predict_with_both_models(args.numeric_model_path, args.context_model_dir, args.file_path, output_path)
        else:
            predict_with_model(args.model_path, args.file_path, output_path)
    
    if args.file_path and args.operation != 'test':
        print(f"Input Data File: {args.file_path}")

# --- Training Functions ---

def train_numeric(model_path, file_path, config_path, gpu_id: int = 0):
    """Train a numeric model"""
    print("Training numeric model...")
    extractor = NumericExtractor(output_path=model_path)
    train_data = load_data(file_path)
    extractor.train(train_data, config_path=config_path, gpu_id=gpu_id)

def train_context(model_path, file_path, config_path, gpu_id: int = 0):
    """Train a context model"""
    print("Training context model...")
    extractor = ContextExtractor(output_path=model_path)
    train_data = load_data(file_path)
    extractor.train(train_data, config_path=config_path, gpu_id=gpu_id)

def train_both(numeric_model_path, context_model_dir, file_path, config_path, gpu_id: int = 0):
    """Train both numeric and context models"""
    print("Training both numeric and context models...")
    train_numeric(numeric_model_path, file_path, config_path, gpu_id=gpu_id)
    train_context(context_model_dir, file_path, config_path, gpu_id=gpu_id)

# --- Evaluation Function (New) ---

def evaluate_files(pred_path, gold_path, numeric_labels_path, match_type='exact', output_path=None, use_timestamp=False):
    """
    Evaluate predictions against gold data using provided files.
    No models are loaded here.
    """
    print("Running evaluation...")
    
    # Load numeric labels
    print(f"Loading numeric labels from {numeric_labels_path}...")
    numeric_labels = Evaluator.load_numeric_labels(numeric_labels_path)
    
    # Load predictions and gold data
    print(f"Loading predictions from {pred_path}...")
    pred_data = Evaluator.load_jsonl(pred_path)
    
    print(f"Loading gold data from {gold_path}...")
    gold_data = Evaluator.load_jsonl(gold_path)
    
    # Initialize evaluator
    print(f"Evaluating with {match_type} matching...")
    evaluator = Evaluator(numeric_labels, match_type=match_type)
    
    # Run evaluation
    results = evaluator.evaluate(pred_data, gold_data) 
    
    # --- Prepare Meta Data and Save Evaluation Results (JSON) ---
    meta_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else None,
        "match_type": match_type,
        "test_data_path": gold_path,
        "prediction_file_path": pred_path,
        "numeric_labels_path": numeric_labels_path,
    }
    
    # Combine results and meta data
    final_results = {
        "meta_data": meta_data,
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    # Print summary to console
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nMatch type: {match_type}")
    print(f"\nOverall Performance:")
    print(f"  Precision: {results['overall']['precision']:.4f}")
    print(f"  Recall:    {results['overall']['recall']:.4f}")
    print(f"  F1 Score:  {results['overall']['f1']:.4f}")
    
    print(f"\nNumeric Labels (Macro Average):")
    print(f"  Precision: {results['numeric_labels']['macro']['precision']:.4f}")
    print(f"  Recall:    {results['numeric_labels']['macro']['recall']:.4f}")
    print(f"  F1 Score:  {results['numeric_labels']['macro']['f1']:.4f}")
    
    print(f"\nNon-numeric Labels (Macro Average):")
    print(f"  Precision: {results['non_numeric_labels']['macro']['precision']:.4f}")
    print(f"  Recall:    {results['non_numeric_labels']['macro']['recall']:.4f}")
    print(f"  F1 Score:  {results['non_numeric_labels']['macro']['f1']:.4f}")
    
    print(f"\nDetailed JSON results saved to: {output_path}")
    print("="*50)
    
    return results

# --- Prediction Functions ---

def predict_with_model(model_path, file_path, output_path=None):
    """Make predictions using a single model"""
    print("Making predictions with single model...")
    print("Single model prediction not fully implemented. Please use combined model prediction with --numeric_model_path and --context_model_dir")

def predict_with_both_models(numeric_model_path, context_model_dir, file_path, output_path):
    """Make predictions using both numeric and context models"""
    print("Making predictions with combined numeric and context models...")
    
    # --- Auto-append model-best if needed ---
    if numeric_model_path and not numeric_model_path.endswith("/model-best"):
        numeric_model_path = os.path.join(numeric_model_path, "model-best")
    # ----------------------------------------

    # CombinedExtractor loads model-best internally, so just pass the directory/base path
    extractor = CombinedExtractor(numeric_model_path=numeric_model_path, 
                                  context_model_path=context_model_dir)
    data = load_data(file_path) 

    predictions = extractor.predict_all(data) 
    save_data(output_path, predictions)
    
    print(f"Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()