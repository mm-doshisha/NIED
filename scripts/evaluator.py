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
    
    def __init__(self, numeric_labels):
        """
        Initialize evaluator.
        
        Args:
            numeric_labels (set): Set of numeric label strings
        """
        self.numeric_labels = numeric_labels
    
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
        
        label2gold = {}
        for g in gold_spans:
            label2gold.setdefault(g[2], []).append(g)
        
        for p in sorted(pred_spans, key=lambda x: (x[2], x[0], x[1])):
            candidates = label2gold.get(p[2], [])
            match_gold = None
            for g in candidates:
                if g in used_gold:
                    continue
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
    
    def compute_numeric_agnostic_metrics_per_doc(self, pred_numeric_spans, gold_numeric_spans, match_type):
        """
        Compute label-agnostic metrics for numeric entities in a single document.
        
        Args:
            pred_numeric_spans (set): Predicted numeric spans
            gold_numeric_spans (set): Gold numeric spans
            match_type (str): 'exact' or 'boundary'
            
        Returns:
            tuple: (tp, fp, fn)
        """
        pred_pos = set((s, e) for (s, e, _) in pred_numeric_spans)
        gold_pos = set((s, e) for (s, e, _) in gold_numeric_spans)
        
        if match_type == 'boundary':
            mapping = self.boundary_pair_match_positions(pred_pos, gold_pos)
            tp = len(mapping)
            fp = len(pred_pos) - tp
            fn = len(gold_pos) - tp
        else:
            matched = pred_pos & gold_pos
            tp = len(matched)
            fp = len(pred_pos) - tp
            fn = len(gold_pos) - tp
        
        return tp, fp, fn
    
    def get_related_non_numeric_spans(self, num_id, relations, id2span):
        """
        Get non-numeric spans related to a numeric entity.
        
        Args:
            num_id: ID of the numeric entity
            relations: List of relations
            id2span: Mapping from entity ID to span
            
        Returns:
            set: Set of related non-numeric spans
        """
        related = set()
        for rel in relations:
            if rel['from_id'] == num_id:
                to_span = id2span.get(rel['to_id'])
                if to_span and to_span[2] not in self.numeric_labels:
                    related.add(to_span)
            elif rel['to_id'] == num_id:
                from_span = id2span.get(rel['from_id'])
                if from_span and from_span[2] not in self.numeric_labels:
                    related.add(from_span)
        return related
    
    def evaluate_document(self, pred_doc, gold_doc, match_type, label_stats=None):
        """
        Evaluate a single document.
        
        Args:
            pred_doc (dict): Predicted document with entities and relations
            gold_doc (dict): Gold document with entities and relations
            match_type (str): 'exact' or 'boundary'
            label_stats (dict): Dictionary to accumulate per-label statistics
            
        Returns:
            tuple: (tp, fp, fn, agnostic_tp, agnostic_fp, agnostic_fn)
        """
        pred_entities = pred_doc.get('entities', [])
        gold_entities = gold_doc.get('entities', [])
        pred_relations = pred_doc.get('relations', [])
        gold_relations = gold_doc.get('relations', [])
        
        pred_id2span, pred_span2id = self.build_span_dict(pred_entities)
        gold_id2span, gold_span2id = self.build_span_dict(gold_entities)
        
        pred_numeric_spans = set([span for span in pred_span2id if span[2] in self.numeric_labels])
        gold_numeric_spans = set([span for span in gold_span2id if span[2] in self.numeric_labels])
        
        # Compute label-agnostic metrics for this document
        ag_tp, ag_fp, ag_fn = self.compute_numeric_agnostic_metrics_per_doc(
            pred_numeric_spans, gold_numeric_spans, match_type
        )
        
        # Match numeric spans
        if match_type == 'boundary':
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
        
        # Evaluate numeric entities - matched
        for span in matched_pred_numeric:
            tp += 1
            if label_stats is not None:
                label = span[2]
                label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                label_stats[label]['tp'] += 1
        
        # Evaluate numeric entities - false positives
        for span in pred_numeric_spans - matched_pred_numeric:
            fp += 1
            if label_stats is not None:
                label = span[2]
                label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                label_stats[label]['fp'] += 1
        
        # Evaluate numeric entities - false negatives
        for span in gold_numeric_spans - matched_gold_numeric:
            fn += 1
            if label_stats is not None:
                label = span[2]
                label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                label_stats[label]['fn'] += 1
        
        # Evaluate related non-numeric entities for MATCHED numeric entities
        for pred_num_span, gold_num_span in num_mapping.items():
            pred_num_id = pred_span2id[pred_num_span]
            gold_num_id = gold_span2id[gold_num_span]
            
            pred_targets = self.get_related_non_numeric_spans(pred_num_id, pred_relations, pred_id2span)
            gold_targets = self.get_related_non_numeric_spans(gold_num_id, gold_relations, gold_id2span)
            
            if match_type == 'boundary':
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
        
        # Evaluate related non-numeric entities for UNMATCHED predicted numeric entities (FP)
        for span in pred_numeric_spans - matched_pred_numeric:
            pred_num_id = pred_span2id[span]
            pred_targets = self.get_related_non_numeric_spans(pred_num_id, pred_relations, pred_id2span)
            
            for target_span in pred_targets:
                fp += 1
                if label_stats is not None:
                    label = target_span[2]
                    label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                    label_stats[label]['fp'] += 1
        
        # Evaluate related non-numeric entities for UNMATCHED gold numeric entities (FN)
        for span in gold_numeric_spans - matched_gold_numeric:
            gold_num_id = gold_span2id[span]
            gold_targets = self.get_related_non_numeric_spans(gold_num_id, gold_relations, gold_id2span)
            
            for target_span in gold_targets:
                fn += 1
                if label_stats is not None:
                    label = target_span[2]
                    label_stats.setdefault(label, {'tp': 0, 'fp': 0, 'fn': 0})
                    label_stats[label]['fn'] += 1
        
        return tp, fp, fn, ag_tp, ag_fp, ag_fn
    
    def _evaluate_with_match_type(self, pred_data, gold_data, match_type):
        """
        Evaluate predictions against gold standard with a specific match type.
        
        Args:
            pred_data (list): List of predicted documents
            gold_data (list): List of gold documents
            match_type (str): 'exact' or 'boundary'
            
        Returns:
            dict: Evaluation results for this match type
        """
        gold_id2doc = {doc['id']: doc for doc in gold_data}
        pred_id2doc = {doc['id']: doc for doc in pred_data}
        all_ids = set(gold_id2doc.keys()) | set(pred_id2doc.keys())
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_ag_tp = 0
        total_ag_fp = 0
        total_ag_fn = 0
        label_stats = {}
        
        for doc_id in all_ids:
            pred_doc = pred_id2doc.get(doc_id, {'entities': [], 'relations': []})
            gold_doc = gold_id2doc.get(doc_id, {'entities': [], 'relations': []})
            tp, fp, fn, ag_tp, ag_fp, ag_fn = self.evaluate_document(pred_doc, gold_doc, match_type, label_stats)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_ag_tp += ag_tp
            total_ag_fp += ag_fp
            total_ag_fn += ag_fn
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        ag_precision = total_ag_tp / (total_ag_tp + total_ag_fp) if (total_ag_tp + total_ag_fp) > 0 else 0.0
        ag_recall = total_ag_tp / (total_ag_tp + total_ag_fn) if (total_ag_tp + total_ag_fn) > 0 else 0.0
        ag_f1 = 2 * ag_precision * ag_recall / (ag_precision + ag_recall) if (ag_precision + ag_recall) > 0 else 0.0
        
        numeric_label_stats = {k: v for k, v in label_stats.items() if k in self.numeric_labels}
        non_numeric_label_stats = {k: v for k, v in label_stats.items() if k not in self.numeric_labels}
        
        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'numeric_agnostic': {
                'precision': ag_precision,
                'recall': ag_recall,
                'f1': ag_f1
            },
            'numeric_labels': self._compute_label_metrics(numeric_label_stats),
            'non_numeric_labels': self._compute_label_metrics(non_numeric_label_stats)
        }
    
    def evaluate(self, pred_data, gold_data, output_path=None):
        """
        Evaluate predictions against gold standard with both exact and boundary matching.
        
        Args:
            pred_data (list): List of predicted documents
            gold_data (list): List of gold documents
            output_path (str): Path to save evaluation results
            
        Returns:
            dict: Evaluation results including both exact and boundary metrics
        """
        results = {
            'exact': self._evaluate_with_match_type(pred_data, gold_data, 'exact'),
            'boundary': self._evaluate_with_match_type(pred_data, gold_data, 'boundary')
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
        
        macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        macro_prec = sum(prec_list) / len(prec_list) if prec_list else 0.0
        macro_rec = sum(rec_list) / len(rec_list) if rec_list else 0.0
        
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
            for match_type in ['exact', 'boundary']:
                f.write(f"{'='*60}\n")
                f.write(f"Match type: {match_type.upper()}\n")
                f.write(f"{'='*60}\n\n")
                
                match_results = results[match_type]
                
                f.write("=== Overall Metrics ===\n")
                f.write(f"Precision: {match_results['overall']['precision']:.4f}\n")
                f.write(f"Recall:    {match_results['overall']['recall']:.4f}\n")
                f.write(f"F1 (micro):{match_results['overall']['f1']:.4f}\n\n")
                
                f.write("=== Numeric (Label-agnostic) ===\n")
                f.write(f"Precision:  {match_results['numeric_agnostic']['precision']:.4f}\n")
                f.write(f"Recall:     {match_results['numeric_agnostic']['recall']:.4f}\n")
                f.write(f"F1 (micro): {match_results['numeric_agnostic']['f1']:.4f}\n\n")
                
                self._print_label_section(f, match_results['numeric_labels'], "Numeric Labels")
                self._print_label_section(f, match_results['non_numeric_labels'], "Non-numeric Labels")
                
                f.write("\n")
    
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