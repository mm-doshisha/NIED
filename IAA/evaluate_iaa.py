#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IAA (Inter-Annotator Agreement) evaluation script for NIED corpus
Calculates pairwise F1 scores between gold and other annotators

Usage example:
  python evaluate_iaa.py
  python evaluate_iaa.py --gold-file data/annotator_gold.jsonl --annotator-files data/annotator_01.jsonl data/annotator_02.jsonl
"""
import argparse
import json
import os
from collections import defaultdict
# datetimeのインポートを削除しました


def load_data_with_text(jsonl_path, target_label=None, numeric_labels=None, filter_non_numeric_unrelated=False):
    """Load text and spans from JSONL file
    
    Args:
        jsonl_path: Path to JSONL file
        target_label: Optional specific label to filter
        numeric_labels: Set of numeric label names
        filter_non_numeric_unrelated: If True, filter non-numeric spans unrelated to numeric ones
    
    Returns:
        List of dicts with 'text' and 'spans' keys
    """
    data_per_doc = []
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '')
            spans = set()

            entities = data.get('entities', [])
            relations = data.get('relations', [])

            if filter_non_numeric_unrelated and numeric_labels is not None:
                # Build entity ID mappings
                id_to_ent = {ent['id']: (ent['start_offset'], ent['end_offset'], ent['label']) 
                            for ent in entities if 'id' in ent}
                id_to_label = {ent['id']: ent['label'] for ent in entities if 'id' in ent}

                # Get numeric entity IDs
                numeric_entity_ids = {eid for eid, label in id_to_label.items() if label in numeric_labels}

                # Get non-numeric entities related to numeric ones
                related_non_numeric_ids = set()
                for rel in relations:
                    from_id = rel.get('from_id')
                    to_id = rel.get('to_id')
                    if from_id in id_to_label and to_id in id_to_label:
                        from_is_num = id_to_label[from_id] in numeric_labels
                        to_is_num = id_to_label[to_id] in numeric_labels
                        if from_is_num and not to_is_num:
                            related_non_numeric_ids.add(to_id)
                        if to_is_num and not from_is_num:
                            related_non_numeric_ids.add(from_id)

                allowed_ids = numeric_entity_ids | related_non_numeric_ids

                for ent in entities:
                    label = ent['label']
                    if (target_label is not None) and (label != target_label):
                        continue
                    if label in numeric_labels:
                        spans.add((ent['start_offset'], ent['end_offset'], label))
                        continue
                    if ent.get('id') in allowed_ids:
                        spans.add((ent['start_offset'], ent['end_offset'], label))
            else:
                for ent in entities:
                    label = ent['label']
                    if (target_label is None) or (label == target_label):
                        spans.add((ent['start_offset'], ent['end_offset'], label))
            
            data_per_doc.append({'text': text, 'spans': spans})
    
    return data_per_doc


def find_common_data(gold_data, annotator_data_list):
    """Find common documents across all files based on text matching
    
    Args:
        gold_data: List of gold annotator documents
        annotator_data_list: List of lists, each containing documents from one annotator
    
    Returns:
        Tuple of (common_gold_data, common_annotator_data_list, common_texts)
    """
    gold_text_to_data = {item['text']: item for item in gold_data}
    
    annotator_text_to_data = []
    for annotator_data in annotator_data_list:
        text_to_data = {item['text']: item for item in annotator_data}
        annotator_text_to_data.append(text_to_data)
    
    # Find texts common to all files
    common_texts = set(gold_text_to_data.keys())
    for text_to_data in annotator_text_to_data:
        common_texts = common_texts.intersection(set(text_to_data.keys()))
    
    # Extract common data
    common_gold_data = [gold_text_to_data[text] for text in common_texts]
    common_annotator_data_list = []
    for text_to_data in annotator_text_to_data:
        common_annotator_data = [text_to_data[text] for text in common_texts]
        common_annotator_data_list.append(common_annotator_data)
    
    return common_gold_data, common_annotator_data_list, list(common_texts)


def exact_match(spans1, spans2):
    """Calculate exact span matches"""
    return spans1 & spans2


def boundary_match(spans1, spans2):
    """Calculate boundary matches (start OR end position matches)"""
    matched = set()
    for s1 in spans1:
        for s2 in spans2:
            if s1[2] == s2[2] and (s1[0] == s2[0] or s1[1] == s2[1]):
                matched.add(s1)
    return matched


def pairwise_f1(spans1_list, spans2_list, match_func):
    """Calculate pairwise F1 score between two annotators
    
    Args:
        spans1_list: List of span sets from annotator 1
        spans2_list: List of span sets from annotator 2
        match_func: Function to determine matching spans
    
    Returns:
        Tuple of (f1, precision, recall, tp, fp, fn)
    """
    tp = fp = fn = 0
    for s1, s2 in zip(spans1_list, spans2_list):
        match = match_func(s1, s2)
        tp += len(match)
        fp += len(s1 - match)
        fn += len(s2 - match)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall, tp, fp, fn


def labelwise_f1(s1_list, s2_list, match_func):
    """Calculate F1 scores per label category
    
    Args:
        s1_list: List of span sets from annotator 1
        s2_list: List of span sets from annotator 2
        match_func: Function to determine matching spans
    
    Returns:
        Dict mapping label to (f1, precision, recall, tp, fp, fn)
    """
    label_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for s1, s2 in zip(s1_list, s2_list):
        match = match_func(s1, s2)
        labels = set([x[2] for x in s1 | s2])
        
        for label in labels:
            s1_label = set([x for x in s1 if x[2] == label])
            s2_label = set([x for x in s2 if x[2] == label])
            m_label = set([x for x in match if x[2] == label])
            
            label_stats[label]['tp'] += len(m_label)
            label_stats[label]['fp'] += len(s1_label - m_label)
            label_stats[label]['fn'] += len(s2_label - m_label)
    
    label_results = {}
    for label, stat in label_stats.items():
        tp, fp, fn = stat['tp'], stat['fp'], stat['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        label_results[label] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return label_results


def load_numeric_labels(numeric_labels_path):
    """Load numeric label names from file"""
    if not os.path.exists(numeric_labels_path):
        # Default numeric labels from the paper
        return {'DATA_COUNT', 'ANNOTATION', 'COLLECTION', 'DOMAIN', 'CONTRIBUTOR'}
    
    with open(numeric_labels_path, encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())


def positions_from_numeric_spans(spans, numeric_labels):
    """Extract (start, end) positions from numeric spans, label-agnostic"""
    return set((s, e) for (s, e, lab) in spans if lab in numeric_labels)


def boundary_pair_match_positions(pos1, pos2):
    """Match positions with boundary overlap (start OR end matches), greedy 1-to-1 mapping
    
    Returns:
        Dict mapping pos1 -> pos2
    """
    mapping = {}
    used = set()
    
    for p in sorted(pos1, key=lambda x: (x[0], x[1])):
        match_g = None
        for g in sorted(pos2, key=lambda x: (x[0], x[1])):
            if g in used:
                continue
            if (p[0] == g[0]) or (p[1] == g[1]):
                match_g = g
                break
        
        if match_g is not None:
            mapping[p] = match_g
            used.add(match_g)
    
    return mapping


def pairwise_numeric_agnostic_metrics(spans1_list, spans2_list, numeric_labels, match_type='exact'):
    """Calculate label-agnostic numeric entity matching metrics
    
    Args:
        spans1_list: List of span sets from annotator 1
        spans2_list: List of span sets from annotator 2
        numeric_labels: Set of numeric label names
        match_type: 'exact' or 'boundary'
    
    Returns:
        Tuple of (f1, precision, recall, tp, fp, fn)
    """
    tp = fp = fn = 0
    
    for s1, s2 in zip(spans1_list, spans2_list):
        p1 = positions_from_numeric_spans(s1, numeric_labels)
        p2 = positions_from_numeric_spans(s2, numeric_labels)
        
        if match_type == 'boundary':
            mapping = boundary_pair_match_positions(p1, p2)
            tp_doc = len(mapping)
            fp_doc = len(p1) - tp_doc
            fn_doc = len(p2) - tp_doc
        else:
            matched = p1 & p2
            tp_doc = len(matched)
            fp_doc = len(p1) - tp_doc
            fn_doc = len(p2) - tp_doc
        
        tp += tp_doc
        fp += fp_doc
        fn += fn_doc
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return f1, precision, recall, tp, fp, fn


def main():
    parser = argparse.ArgumentParser(
        description='Calculate IAA pairwise F1 scores between gold and other annotators'
    )
    parser.add_argument(
        '--gold-file',
        default='data/annotator_gold.jsonl',
        help='Path to gold annotator JSONL file'
    )
    parser.add_argument(
        '--annotator-files',
        nargs='+',
        default=['data/annotator_01.jsonl', 'data/annotator_02.jsonl', 'data/annotator_03.jsonl'],
        help='Paths to annotator JSONL files'
    )
    parser.add_argument(
        '--output-dir',
        default='result',
        help='Output directory for results'
    )
    parser.add_argument(
        '--match-type',
        nargs='+',
        default=['exact', 'boundary'],
        choices=['exact', 'boundary'],
        help='Matching methods to evaluate'
    )
    parser.add_argument(
        '--numeric-labels-file',
        default=None,
        help='Path to numeric labels file (one label per line)'
    )
    parser.add_argument(
        '--target-label',
        default=None,
        help='Evaluate only a specific label'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify files exist
    if not os.path.exists(args.gold_file):
        print(f"Error: Gold file not found: {args.gold_file}")
        return
    
    for ann_file in args.annotator_files:
        if not os.path.exists(ann_file):
            print(f"Error: Annotator file not found: {ann_file}")
            return
    
    # Load numeric labels
    if args.numeric_labels_file:
        numeric_labels = load_numeric_labels(args.numeric_labels_file)
    else:
        numeric_labels = {'DATA_COUNT', 'ANNOTATION', 'COLLECTION', 'DOMAIN', 'CONTRIBUTOR'}
    
    # Load data
    print(f"Loading gold file: {args.gold_file}")
    gold_data = load_data_with_text(
        args.gold_file,
        args.target_label,
        numeric_labels,
        filter_non_numeric_unrelated=True
    )
    
    annotator_data_list = []
    annotator_names = []
    for ann_file in args.annotator_files:
        print(f"Loading annotator file: {ann_file}")
        ann_data = load_data_with_text(
            ann_file,
            args.target_label,
            numeric_labels,
            filter_non_numeric_unrelated=True
        )
        annotator_data_list.append(ann_data)
        annotator_names.append(os.path.splitext(os.path.basename(ann_file))[0])
    
    # Find common documents
    common_gold_data, common_annotator_data_list, common_texts = find_common_data(
        gold_data,
        annotator_data_list
    )
    
    print(f"\nGold file total documents: {len(gold_data)}")
    print(f"Common documents found: {len(common_gold_data)}")
    
    if len(common_gold_data) == 0:
        print("Error: No common documents found")
        return
    
    # Extract spans
    gold_spans = [item['spans'] for item in common_gold_data]
    annotator_spans_list = []
    for annotator_data in common_annotator_data_list:
        annotator_spans = [item['spans'] for item in annotator_data]
        annotator_spans_list.append(annotator_spans)
    
    # Evaluate for each match type
    for match_type in args.match_type:
        print(f"\n{'='*60}")
        print(f"Evaluating with {match_type} matching")
        print('='*60)
        
        match_func = exact_match if match_type == 'exact' else boundary_match
        
        # Initialize results structure with metadata first
        # 変更点: evaluation_dateを出力しないように削除しました
        results = {
            'metadata': {
                'match_type': match_type,
                'gold_file': args.gold_file,
                'annotator_files': args.annotator_files,
                'num_common_documents': len(common_gold_data),
                'target_label': args.target_label
            }
        }
        
        # Collect all pairwise results first to calculate summary
        pairwise_results = []
        all_f1s = []
        all_label_results = defaultdict(list)
        all_numeric_agnostic = []
        
        # Evaluate each annotator against gold
        for i, (ann_name, ann_spans) in enumerate(zip(annotator_names, annotator_spans_list)):
            print(f"\nEvaluating: gold vs {ann_name}")
            
            # Overall F1
            f1, precision, recall, tp, fp, fn = pairwise_f1(gold_spans, ann_spans, match_func)
            all_f1s.append(f1)
            
            # Label-wise F1
            label_results = labelwise_f1(gold_spans, ann_spans, match_func)
            for label, metrics in label_results.items():
                all_label_results[label].append(metrics['f1'])
            
            # Numeric label-agnostic detection
            num_f1, num_p, num_r, num_tp, num_fp, num_fn = pairwise_numeric_agnostic_metrics(
                gold_spans,
                ann_spans,
                numeric_labels,
                match_type
            )
            all_numeric_agnostic.append(num_f1)
            
            pair_result = {
                'annotator': ann_name,
                'overall': {
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                },
                'numeric_agnostic': {
                    'f1': num_f1,
                    'precision': num_p,
                    'recall': num_r,
                    'tp': num_tp,
                    'fp': num_fp,
                    'fn': num_fn
                },
                'per_label': {}
            }
            
            # Separate numeric and non-numeric labels
            numeric_label_results = {}
            non_numeric_label_results = {}
            
            for label, metrics in label_results.items():
                if label in numeric_labels:
                    numeric_label_results[label] = metrics
                else:
                    non_numeric_label_results[label] = metrics
            
            pair_result['per_label']['numeric'] = numeric_label_results
            pair_result['per_label']['non_numeric'] = non_numeric_label_results
            
            pairwise_results.append(pair_result)
            
            print(f"  Overall F1: {f1:.4f}")
            print(f"  Numeric agnostic F1: {num_f1:.4f}")
        
        # Calculate summary statistics
        summary = {
            'average_f1': sum(all_f1s) / len(all_f1s) if all_f1s else 0.0,
            'average_numeric_agnostic_f1': sum(all_numeric_agnostic) / len(all_numeric_agnostic) if all_numeric_agnostic else 0.0,
            'average_per_label': {}
        }
        
        # Average per label
        numeric_avg = {}
        non_numeric_avg = {}
        
        for label, f1_list in all_label_results.items():
            avg = sum(f1_list) / len(f1_list) if f1_list else 0.0
            if label in numeric_labels:
                numeric_avg[label] = avg
            else:
                non_numeric_avg[label] = avg
        
        summary['average_per_label']['numeric'] = numeric_avg
        summary['average_per_label']['non_numeric'] = non_numeric_avg
        
        # Calculate micro and macro averages
        for label_type in ['numeric', 'non_numeric']:
            total_tp = total_fp = total_fn = 0
            label_f1s = []
            
            for pair_result in pairwise_results:
                for label, metrics in pair_result['per_label'][label_type].items():
                    total_tp += metrics['tp']
                    total_fp += metrics['fp']
                    total_fn += metrics['fn']
                    label_f1s.append(metrics['f1'])
            
            micro = (2 * total_tp / (2 * total_tp + total_fp + total_fn)) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0
            macro = sum(label_f1s) / len(label_f1s) if label_f1s else 0.0
            
            summary['average_per_label'][f'{label_type}_micro'] = micro
            summary['average_per_label'][f'{label_type}_macro'] = macro
        
        # Add summary before pairwise_results
        results['summary'] = summary
        results['pairwise_results'] = pairwise_results
        
        # Save results
        output_path = os.path.join(args.output_dir, f'iaa_results_{match_type}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Average F1: {results['summary']['average_f1']:.4f}")
        print(f"Average numeric agnostic F1: {results['summary']['average_numeric_agnostic_f1']:.4f}")


if __name__ == '__main__':
    main()