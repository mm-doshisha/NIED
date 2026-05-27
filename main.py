import argparse
import sys
import os
import time
import json
from datetime import datetime
from tqdm import tqdm

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

def load_numeric_labels(filepath: str) -> set:
    """Load numeric labels from file."""
    labels = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            label = line.strip()
            if label:
                labels.add(label)
    return labels

def main():
    parser = argparse.ArgumentParser(description='ML model operations script')
    
    parser.add_argument('--operation', required=True, choices=['train', 'test', 'predict'], 
                        help='Operation to perform')
    
    parser.add_argument('--type', choices=['numeric', 'context', 'both'], default='both',
                        help='Type of training (Default: both)')
    
    parser.add_argument('--config_path', default='config/config_roberta-base.cfg', 
                        help='Path to the spaCy training config file (e.g., config/config_roberta-base.cfg)')
    
    parser.add_argument('--use_timestamp', action='store_true', 
                        help='Use a timestamp suffix in the default output/model paths (Default: False)')

    parser.add_argument('--model_path', help='Path to the model (for single model operations)')
    parser.add_argument('--numeric_model_path', help='Path to the numeric model')
    parser.add_argument('--context_model_dir', help='Directory path to the context model')
    
    parser.add_argument('--file_path', help='Path to the input data file (Required for train/predict)')
    parser.add_argument('--pred_path', help='Path to the prediction file (Required for test operation)')
    parser.add_argument('--gold_path', help='Path to the gold/correct answer file (Required for test operation)')
    
    parser.add_argument('--use_gold_numeric', action='store_true',
                        help='Use gold numeric entities from input file for context-only prediction')

    parser.add_argument('--output_path', help='Path to save prediction/evaluation results')
    
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use (spaCy --gpu-id)')
    
    parser.add_argument('--numeric_labels', default='labels/numeric_labels.txt', 
                        help='Path to numeric labels file (for test operation)')
    
    args = parser.parse_args()
    
    suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if args.use_timestamp else ""
    
    default_config_name = os.path.basename(args.config_path).replace('config_', '').replace('.cfg', '')

    if args.operation == 'train':
        if not args.file_path:
            print("Error: Train operation requires --file_path")
            sys.exit(1)
            
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
        
        using_gold_numeric = args.use_gold_numeric
        using_combined_models = args.numeric_model_path and args.context_model_dir
        
        if not using_gold_numeric and not using_combined_models and not args.model_path:
            print("Warning: No model path provided for predict operation. This will likely fail.")
        
        if using_gold_numeric and not args.context_model_dir:
            print("Error: --use_gold_numeric requires --context_model_dir")
            sys.exit(1)

    print(f"Operation: {args.operation}")

    if args.operation == 'train':
        print(f"Training type: {args.type}")
        if args.type == 'both':
            os.makedirs(args.context_model_dir, exist_ok=True)
            train_both(args.numeric_model_path, args.context_model_dir, args.file_path, args.config_path, gpu_id=args.gpu_id)
        else:
            if args.type == 'numeric':
                train_numeric(args.model_path, args.file_path, args.config_path, gpu_id=args.gpu_id)
            elif args.type == 'context':
                train_context(args.model_path, args.file_path, args.config_path, gpu_id=args.gpu_id)
    
    elif args.operation == 'test':
        pred_basename = os.path.splitext(os.path.basename(args.pred_path))[0]
        
        output_suffix = suffix if args.use_timestamp else ""
        default_output_path = f"evaluations/evaluation_{pred_basename}{output_suffix}.json"
        
        output_path = args.output_path if args.output_path else default_output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Evaluation results path: {output_path}")
        
        evaluate_files(
            pred_path=args.pred_path,
            gold_path=args.gold_path,
            numeric_labels_path=args.numeric_labels,
            output_path=output_path,
            use_timestamp=args.use_timestamp
        )
    
    elif args.operation == 'predict':
        
        using_gold_numeric = args.use_gold_numeric
        using_combined_models = args.numeric_model_path and args.context_model_dir
        
        if using_gold_numeric and args.context_model_dir:
            ctx_core = get_core_model_name(args.context_model_dir, "context_model_")
            model_identifier = f"gold-numeric_ctx-{ctx_core}"
        elif using_combined_models:
            num_core = get_core_model_name(args.numeric_model_path, "numeric_model_")
            ctx_core = get_core_model_name(args.context_model_dir, "context_model_")
            
            if num_core == ctx_core:
                model_identifier = num_core
            else:
                model_identifier = f"num-{num_core}_ctx-{ctx_core}"
        else:
            model_identifier = os.path.basename(args.model_path) if args.model_path else "single"
        
        output_suffix = suffix if args.use_timestamp else ""
        default_output_path = f"predictions/predictions_{model_identifier}{output_suffix}.jsonl"
        
        output_path = args.output_path if args.output_path else default_output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Prediction output path: {output_path}")
        
        if using_gold_numeric and args.context_model_dir:
            predict_with_gold_numeric(
                file_path=args.file_path,
                context_model_dir=args.context_model_dir,
                output_path=output_path,
                numeric_labels_path=args.numeric_labels
            )
        elif using_combined_models:
            predict_with_both_models(args.numeric_model_path, args.context_model_dir, args.file_path, output_path)
        else:
            predict_with_model(args.model_path, args.file_path, output_path)
    
    if args.file_path and args.operation != 'test':
        print(f"Input Data File: {args.file_path}")


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


def evaluate_files(pred_path, gold_path, numeric_labels_path, output_path=None, use_timestamp=False):
    """Evaluate predictions against gold data using provided files."""
    print("Running evaluation...")
    
    print(f"Loading numeric labels from {numeric_labels_path}...")
    numeric_labels = Evaluator.load_numeric_labels(numeric_labels_path)
    
    print(f"Loading predictions from {pred_path}...")
    pred_data = Evaluator.load_jsonl(pred_path)
    
    print(f"Loading gold data from {gold_path}...")
    gold_data = Evaluator.load_jsonl(gold_path)
    
    print("Evaluating with exact and boundary matching...")
    evaluator = Evaluator(numeric_labels)
    
    results = evaluator.evaluate(pred_data, gold_data) 
    
    metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S") if use_timestamp else None,
        "test_data_path": gold_path,
        "prediction_file_path": pred_path,
        "numeric_labels_path": numeric_labels_path,
    }
    
    final_results = {
        "metadata": metadata,
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for match_type in ['exact', 'boundary']:
        match_results = results[match_type]
        print(f"\n{'='*60}")
        print(f"Match type: {match_type.upper()}")
        print(f"{'='*60}")
        
        print(f"\nOverall Performance:")
        print(f"  Precision: {match_results['overall']['precision']:.4f}")
        print(f"  Recall:    {match_results['overall']['recall']:.4f}")
        print(f"  F1 Score:  {match_results['overall']['f1']:.4f}")
        
        print(f"\nNumeric Labels (Macro Average):")
        print(f"  Precision: {match_results['numeric_labels']['macro']['precision']:.4f}")
        print(f"  Recall:    {match_results['numeric_labels']['macro']['recall']:.4f}")
        print(f"  F1 Score:  {match_results['numeric_labels']['macro']['f1']:.4f}")
        
        print(f"\nNon-numeric Labels (Macro Average):")
        print(f"  Precision: {match_results['non_numeric_labels']['macro']['precision']:.4f}")
        print(f"  Recall:    {match_results['non_numeric_labels']['macro']['recall']:.4f}")
        print(f"  F1 Score:  {match_results['non_numeric_labels']['macro']['f1']:.4f}")
    
    print(f"\nDetailed JSON results saved to: {output_path}")
    print("="*60)
    
    return results


def predict_with_model(model_path, file_path, output_path=None):
    """Make predictions using a single model"""
    print("Making predictions with single model...")
    print("Single model prediction not fully implemented. Please use combined model prediction with --numeric_model_path and --context_model_dir")

def predict_with_both_models(numeric_model_path, context_model_dir, file_path, output_path):
    """Make predictions using both numeric and context models"""
    print("Making predictions with combined numeric and context models...")
    
    if numeric_model_path and not numeric_model_path.endswith("/model-best"):
        numeric_model_path = os.path.join(numeric_model_path, "model-best")

    extractor = CombinedExtractor(numeric_model_path=numeric_model_path, 
                                  context_model_path=context_model_dir)
    data = load_data(file_path) 

    predictions = extractor.predict_all(data) 
    save_data(output_path, predictions)
    
    print(f"Predictions saved to: {output_path}")

def predict_with_gold_numeric(file_path, context_model_dir, output_path, numeric_labels_path='labels/numeric_labels.txt'):
    """Make predictions using gold numeric entities and context model only"""
    print("Making predictions with gold numeric entities + context model...")
    print(f"  Input file (with gold numeric): {file_path}")
    print(f"  Context model: {context_model_dir}")
    
    numeric_label_set = load_numeric_labels(numeric_labels_path)
    
    gold_data = load_gold_data(file_path)
    
    context_extractor = ContextExtractor(model=context_model_dir)
    
    predictions = []
    for gold_item in tqdm(gold_data, desc="Predicting with gold numeric"):
        doc_id = gold_item.get('id')
        text = gold_item.get('text', '')
        gold_entities = gold_item.get('entities', [])
        gold_relations = gold_item.get('relations', [])
        
        predicted_entities, predicted_relations = context_extractor.predict_with_gold_numeric(
            text=text,
            gold_entities=gold_entities,
            gold_relations=gold_relations,
            numeric_label_set=numeric_label_set
        )
        
        predictions.append({
            'id': doc_id,
            'text': text,
            'entities': predicted_entities,
            'relations': predicted_relations
        })
    
    save_data(output_path, predictions)
    print(f"Predictions saved to: {output_path}")

def load_gold_data(file_path):
    """Load gold data from JSONL file without processing."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

if __name__ == "__main__":
    main()