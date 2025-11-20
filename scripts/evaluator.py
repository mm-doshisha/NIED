#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluator for numeric entity extraction with related non-numeric entities.
Supports both exact and boundary span matching.
"""
import json
from collections import defaultdict


class Evaluator:
    """
    Evaluates predictions against gold standard annotations.
    Supports both exact and boundary matching for span evaluation.
    """
    
    def __init__(self, numeric_labels, match_type='exact'):
        """
        Initialize evaluator.
        
        Args:
            numeric_labels (set): Set of numeric label strings
            match_type (str): 'exact' or 'boundary'
        """
        self.numeric_labels = numeric_labels
        self.match_type = match_type
    
    @staticmethod
    def load_numeric_labels(path):
        """Load numeric labels from file."""
        with open(path, encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    
    @staticmethod
    def load_jsonl(path):
        """Load JSONL file."""
        with open(path, encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    @staticmethod
    def build_span_dict(entities):
        """
        Build bidirectional mapping between entity IDs and spans.
        
        Args:
            entities (list): List of entity dictionaries
            
        Returns:
            tuple: (id2span dict, span2id dict)
        """
        id2span = {}
        span2id = {}
        for ent in entities:
            span = (ent['start_offset'], ent['end_offset'], ent['label'])
            id2span[ent['id']] = span
            span2id[span] = ent['id']
        return id2span, span2id
    
    @staticmethod
    def boundary_pair_match(pred_spans, gold_spans):
        """
        Create greedy 1-to-1 mapping using boundary matching.
        Boundary match: same label AND (same start OR same end).
        
        Args:
            pred_spans (iterable): Predicted spans (start, end, label)
            gold_spans (iterable): Gold spans (start, end, label)
            
        Returns:
            dict: Mapping from pred_span to gold_span
        """
        mapping = {}
        used_gold = set()
        
        # Group gold spans by label
        label2gold = {}
        for g in gold_spans:
            label2gold.setdefault(g[2], []).append(g)
        
        # Process predictions in deterministic order
        for p in sorted(pred_spans, key=lambda x: (x[2], x[0], x[1])):
            candidates = label2gold.get(p[2], [])
            match_gold = None
            for g in candidates:
                if g in used_gold:
                    continue
                # Boundary match: same start OR same end
                if (p[0] == g[0]) or (p[1] == g[1]):
                    match_gold = g
                    break
            if match_gold is not None:
                mapping[p] = match_gold
                used_gold.add(match_gold)
        return mapping
    
    @staticmethod
    def boundary_pair_match_positions(pred_positions, gold_positions):
        """
        Create greedy 1-to-1 mapping using boundary matching (label-agnostic).
        
        Args:
            pred_positions (iterable): Predicted positions (start, end)
            gold_positions (iterable): Gold positions (start, end)
            
        Returns:
            dict: Mapping from pred_pos to gold_pos
        """
        mapping = {}
        used_gold = set()
        
        for p in sorted(pred_positions, key=lambda x: (x[0], x[1])):
            match_g = None
            for g in sorted(gold_positions, key=lambda x: (x[0], x[1])):
                if g in used_gold:
                    continue
                if (p[0] == g[0]) or (p[1] == g[1]):
                    match_g = g
                    break
            if match_g is not None:
                mapping[p] = match_g
                used_gold.add(match_g)
        return mapping
    
    def compute_numeric_agnostic_metrics(self, pred_numeric_spans, gold_numeric_spans):
        """
        Compute label-agnostic metrics for numeric entities.
        
        Args:
            pred_numeric_spans (set): Predicted numeric spans
            gold_numeric_spans (set): Gold numeric spans
            
        Returns:
            tuple: (precision, recall, f1)
        """
        pred_pos = set((s, e) for (s, e, _) in pred_numeric_spans)
        gold_pos = set((s, e) for (s, e, _) in gold_numeric_spans)
        
        if self.match_type == 'boundary':
            mapping = self.boundary_pair_match_positions(pred_pos, gold_pos)
            tp = len(mapping)
            fp = len(pred_pos) - tp
            fn = len(gold_pos) - tp
        else:
            matched = pred_pos & gold_pos
            tp = len(matched)
            fp = len(pred_pos) - tp
            fn = len(gold_pos) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1
    
    def evaluate_document(self, pred_doc, gold_doc, label_stats=None):
        """
        Evaluate a single document.
        
        Args:
            pred_doc (dict): Predicted document with entities and relations
            gold_doc (dict): Gold document with entities and relations
            label_stats (dict): Dictionary to accumulate per-label statistics
            
        Returns:
            tuple: (tp, fp, fn)
        """
        pred_entities = pred_doc.get('entities', [])
        gold_entities = gold_doc.get('entities', [])
        pred_relations = pred_doc.get('relations', [])
        gold_relations = gold_doc.get('relations', [])
        
        pred_id2span, pred_span2id = self.build_span_dict(pred_entities)
        gold_id2span, gold_span2id = self.build_span_dict(gold_entities)
        
        pred_numeric_spans = set([span for span in pred_span2id if span[2] in self.numeric_labels])
        gold_numeric_spans = set([span for span in gold_span2id if span[2] in self.numeric_labels])
        
        # Match numeric spans
        if self.match_type == 'boundary':
            num_mapping = self.boundary_pair_match(pred_numeric_spans, gold_numeric_spans)
            matched_pred_numeric = set(num_mapping.keys())
            matched_gold_numeric = set(num_mapping.values())
        else:
            matched = pred_numeric_spans & gold_numeric_spans
            num_mapping = {s: s for s in matched}
            matched_pred_numeric = set(num_mapping.keys())
            matched_gold_numeric = set(num_mapping.values())
        
        tp = 0
        fp = 0
        fn = 0
        
        # Evaluate numeric entities
        for span in matched_pred_numeric:
            tp += 1
            if label_stats is not None:
                label = span[2]
                label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                label_stats[label]['tp'] += 1
        
        for span in pred_numeric_spans - matched_pred_numeric:
            fp += 1
            if label_stats is not None:
                label = span[2]
                label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                label_stats[label]['fp'] += 1
        
        for span in gold_numeric_spans - matched_gold_numeric:
            fn += 1
            if label_stats is not None:
                label = span[2]
                label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                label_stats[label]['fn'] += 1
        
        # Evaluate related non-numeric entities
        if len(matched_pred_numeric) == 0:
            # Handle unmatched numeric entities
            for span in gold_numeric_spans - matched_gold_numeric:
                num_id = gold_span2id[span]
                for rel in gold_relations:
                    if rel['from_id'] == num_id:
                        to_span = gold_id2span.get(rel['to_id'])
                        if to_span and to_span[2] not in self.numeric_labels:
                            fn += 1
                            if label_stats is not None:
                                label = to_span[2]
                                label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                                label_stats[label]['fn'] += 1
            
            for span in pred_numeric_spans - matched_pred_numeric:
                num_id = pred_span2id[span]
                for rel in pred_relations:
                    if rel['from_id'] == num_id:
                        to_span = pred_id2span.get(rel['to_id'])
                        if to_span and to_span[2] not in self.numeric_labels:
                            fp += 1
                            if label_stats is not None:
                                label = to_span[2]
                                label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                                label_stats[label]['fp'] += 1
            return tp, fp, fn
        
        # For matched numeric entities, evaluate their related non-numeric entities
        for pred_num_span, gold_num_span in num_mapping.items():
            pred_num_id = pred_span2id[pred_num_span]
            gold_num_id = gold_span2id[gold_num_span]
            
            pred_targets = set([pred_id2span[rel['to_id']] for rel in pred_relations 
                               if rel['from_id'] == pred_num_id and pred_id2span[rel['to_id']][2] not in self.numeric_labels])
            gold_targets = set([gold_id2span[rel['to_id']] for rel in gold_relations 
                               if rel['from_id'] == gold_num_id and gold_id2span[rel['to_id']][2] not in self.numeric_labels])
            
            if self.match_type == 'boundary':
                tgt_mapping = self.boundary_pair_match(pred_targets, gold_targets)
                matched_pred_targets = set(tgt_mapping.keys())
                matched_gold_targets = set(tgt_mapping.values())
            else:
                matched = pred_targets & gold_targets
                matched_pred_targets = set(matched)
                matched_gold_targets = set(matched)
            
            for span in matched_pred_targets:
                tp += 1
                if label_stats is not None:
                    label = span[2]
                    label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                    label_stats[label]['tp'] += 1
            
            for span in pred_targets - matched_pred_targets:
                fp += 1
                if label_stats is not None:
                    label = span[2]
                    label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                    label_stats[label]['fp'] += 1
            
            for span in gold_targets - matched_gold_targets:
                fn += 1
                if label_stats is not None:
                    label = span[2]
                    label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                    label_stats[label]['fn'] += 1
        
        return tp, fp, fn
    
    def evaluate(self, pred_data, gold_data, output_path=None):
        """
        Evaluate predictions against gold standard.
        
        Args:
            pred_data (list): List of predicted documents
            gold_data (list): List of gold documents
            output_path (str): Path to save evaluation results
            
        Returns:
            dict: Evaluation results including precision, recall, F1
        """
        gold_text2doc = {doc['text']: doc for doc in gold_data}
        pred_text2doc = {doc['text']: doc for doc in pred_data}
        all_texts = set(gold_text2doc.keys()) | set(pred_text2doc.keys())
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        label_stats = {}
        
        for text in all_texts:
            pred_doc = pred_text2doc.get(text, {'entities': [], 'relations': []})
            gold_doc = gold_text2doc.get(text, {'entities': [], 'relations': []})
            tp, fp, fn = self.evaluate_document(pred_doc, gold_doc, label_stats)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Separate numeric and non-numeric label stats
        numeric_label_stats = {k: v for k, v in label_stats.items() if k in self.numeric_labels}
        non_numeric_label_stats = {k: v for k, v in label_stats.items() if k not in self.numeric_labels}
        
        # Compute label-agnostic numeric metrics
        pred_numeric_spans_all = set()
        gold_numeric_spans_all = set()
        for text in all_texts:
            pred_doc = pred_text2doc.get(text, {'entities': [], 'relations': []})
            gold_doc = gold_text2doc.get(text, {'entities': [], 'relations': []})
            _, pred_span2id = self.build_span_dict(pred_doc.get('entities', []))
            _, gold_span2id = self.build_span_dict(gold_doc.get('entities', []))
            pred_numeric_spans_all |= set([span for span in pred_span2id if span[2] in self.numeric_labels])
            gold_numeric_spans_all |= set([span for span in gold_span2id if span[2] in self.numeric_labels])
        
        ag_prec, ag_rec, ag_f1 = self.compute_numeric_agnostic_metrics(pred_numeric_spans_all, gold_numeric_spans_all)
        
        results = {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'numeric_agnostic': {
                'precision': ag_prec,
                'recall': ag_rec,
                'f1': ag_f1
            },
            'numeric_labels': self._compute_label_metrics(numeric_label_stats),
            'non_numeric_labels': self._compute_label_metrics(non_numeric_label_stats)
        }
        
        if output_path:
            self._print_results(results, output_path)
        
        return results
    
    @staticmethod
    def _compute_label_metrics(label_stats):
        """Compute per-label and aggregate metrics."""
        label_metrics = {}
        f1_list = []
        prec_list = []
        rec_list = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for label, stat in label_stats.items():
            tp = stat['tp']
            fp = stat['fp']
            fn = stat['fn']
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            label_metrics[label] = {
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            
            f1_list.append(f1)
            prec_list.append(prec)
            rec_list.append(rec)
        
        # Macro averages
        macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        macro_prec = sum(prec_list) / len(prec_list) if prec_list else 0.0
        macro_rec = sum(rec_list) / len(rec_list) if rec_list else 0.0
        
        # Micro averages
        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
        
        return {
            'labels': label_metrics,
            'macro': {
                'precision': macro_prec,
                'recall': macro_rec,
                'f1': macro_f1
            },
            'micro': {
                'precision': micro_prec,
                'recall': micro_rec,
                'f1': micro_f1
            }
        }
    
    def _print_results(self, results, output_path):
        """Print evaluation results to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Match type: {self.match_type}\n\n")
            
            # Overall metrics
            f.write("=== Overall Metrics ===\n")
            f.write(f"Precision: {results['overall']['precision']:.4f}\n")
            f.write(f"Recall:    {results['overall']['recall']:.4f}\n")
            f.write(f"F1 (micro):{results['overall']['f1']:.4f}\n\n")
            
            # Numeric label-agnostic metrics
            f.write("=== Numeric (Label-agnostic) ===\n")
            f.write(f"Precision:  {results['numeric_agnostic']['precision']:.4f}\n")
            f.write(f"Recall:     {results['numeric_agnostic']['recall']:.4f}\n")
            f.write(f"F1 (micro): {results['numeric_agnostic']['f1']:.4f}\n\n")
            
            # Numeric labels
            self._print_label_section(f, results['numeric_labels'], "Numeric Labels")
            
            # Non-numeric labels
            self._print_label_section(f, results['non_numeric_labels'], "Non-numeric Labels")
    
    @staticmethod
    def _print_label_section(f, metrics, title):
        """Print metrics for a label category."""
        f.write(f"=== {title} ===\n")
        f.write(f"{'Label':20s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}   TP   FP   FN\n")
        
        for label, stats in sorted(metrics['labels'].items()):
            f.write(f"{label:20s} {stats['precision']:8.4f} {stats['recall']:8.4f} "
                   f"{stats['f1']:8.4f}  {stats['tp']:4d} {stats['fp']:4d} {stats['fn']:4d}\n")
        
        f.write(f"\nMacro Average:\n")
        f.write(f"  Precision: {metrics['macro']['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['macro']['recall']:.4f}\n")
        f.write(f"  F1:        {metrics['macro']['f1']:.4f}\n")
        
        f.write(f"\nMicro Average:\n")
        f.write(f"  Precision: {metrics['micro']['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics['micro']['recall']:.4f}\n")
        f.write(f"  F1:        {metrics['micro']['f1']:.4f}\n\n")