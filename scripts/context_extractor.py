import json
import random
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from spacy.symbols import ORTH
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import subprocess
import os
from scripts.data_processor import DataProcessor

class ContextExtractor:
    NUM_TOKEN_START = "[NUM] "
    NUM_TOKEN_END = " [/NUM]"
    
    def __init__(self, output_path: str = None, model: str = None):
        """
        Initialize the NER model for extracting context entities.
        
        Args:
            output_path: Directory to save the models (Used for train)
            model: Path to the model directory (Used for test/predict)
        """
        
        target_path = output_path if output_path is not None else model

        if target_path is None:
            # If neither model path nor output path is provided, stop execution.
            raise ValueError(
                "ContextExtractor requires either 'model' path (for prediction/testing) "
                "or 'output_path' (for training). None was provided."
            )

        self.output_path = Path(target_path)
        
        # Create output directory if it doesn't exist (needed for training/output saving)
        os.makedirs(self.output_path, exist_ok=True)
        
        self.models = {} # Dictionary to hold models for each label
        
        # Initialize labels using DataProcessor
        self.processor = DataProcessor("labels/all_labels.txt", "labels/numeric_labels.txt")
        self.labels = list(self.processor.valid_labels - self.processor.value_labels)
        
        if not self.labels:
            print("Warning: No non-numeric labels found")
            self.labels = ["UNIT", "QUALIFIER"]  # Default labels
            
        # Load model if specified (for predict/test)
        if model:
            model_path = Path(model)
            if model_path.exists():
                # Search for all model_[label] directories within the path
                for label_dir in model_path.glob("model_*"):
                    if label_dir.is_dir():
                        label = label_dir.name.replace("model_", "")
                        model_best_path = label_dir / "model-best"
                        if model_best_path.exists():
                            try:
                                self.models[label] = spacy.load(str(model_best_path))
                                print(f"Loaded model for label: {label} from {model_best_path}")
                            except Exception as e:
                                # If a label-specific model fails to load, raise an error.
                                raise OSError(f"Error loading context model for label {label} from {model_best_path}. Cannot proceed.") from e
                        else:
                            raise FileNotFoundError(f"Model-best directory not found for label {label} at {model_best_path}. Cannot proceed.")
            else:
                raise FileNotFoundError(f"Context model path {model} does not exist. Cannot proceed.")
    
    def add_custom_tokenizer(self, nlp):
        """
        Add custom tokenizer rules for handling [NUM] tags.
        
        Args:
            nlp: spaCy language model
        """
        tokenizer = nlp.tokenizer
        
        # Define special tokens
        special_cases = {
            "[NUM]": [{ORTH: "[NUM]"}],
            "[/NUM]": [{ORTH: "[/NUM]"}]
        }
        
        # Add special cases to the tokenizer
        for case, tokens in special_cases.items():
            tokenizer.add_special_case(case, tokens)
        
        return tokenizer
    
    def prepare_training_data(self, train_data: List[Dict], label: str):
        """
        Convert processed documents to spaCy training examples for a specific label.
        
        Args:
            train_data: List of training examples
            label: Specific label to prepare data for
        """
        doc_bin = DocBin()
        
        for training_example in tqdm(train_data): 
            text = training_example['text']
            labels = training_example['entities']
            
            # Note: self.nlp must be initialized (e.g., spacy.blank("en")) before this call
            doc = self.nlp.make_doc(text) 
            ents = []
            for label_example in labels:
                if label_example['label'] != label:
                    continue
                    
                start = label_example['start_offset']
                end = label_example['end_offset']
                
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
                if span is None:
                    print("Skipping entity")
                else:
                    ents.append(span)
                    
            filtered_ents = filter_spans(ents)
            doc.ents = filtered_ents 
            doc_bin.add(doc)
        
        # Save to a separate file for each label
        os.makedirs("spacy", exist_ok=True)
        doc_bin.to_disk(f"spacy/train_{label}.spacy")
    
    def train(self, train_data: List[Dict], dev_examples: Optional[List[Dict]] = None, 
               config_path = "config/config_roberta-base.cfg", gpu_id = 0):
        """
        Train the NER model for each context label.
        
        Args:
            train_data: List of training examples
            dev_examples: List of development examples for evaluation
            config_path: Path of the finetuning config file
            gpu_id: GPU ID to use
        """
        train_data = self.convert_data(train_data)
        
        # Train a model for each label
        for label in self.labels:
            print(f"Training model for label: {label}")
            
            # Create a new spaCy model
            self.nlp = spacy.blank("en")
            self.add_custom_tokenizer(self.nlp)
            
            # Add NER component
            if "ner" not in self.nlp.pipe_names:
                self.ner = self.nlp.add_pipe("ner", last=True)
            else:
                self.ner = self.nlp.get_pipe("ner")
            
            # Add label
            self.ner.add_label(label)
            
            # Prepare training data
            self.prepare_training_data(train_data, label)
            subprocess.run("export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0", shell=True, check=True)

            # Set model output path
            model_output_path = os.path.join(self.output_path, f"model_{label}")
            os.makedirs(model_output_path, exist_ok=True)

            # Train the model
            if dev_examples == None:
                subprocess.run(f"python -m spacy train {config_path} --output-path {model_output_path} --paths.train spacy/train_{label}.spacy --paths.dev spacy/train_{label}.spacy --gpu-id {gpu_id}", shell=True, check=True)

            # Save the trained model
            model_path = os.path.join(model_output_path, "model-best")
            if os.path.exists(model_path):
                self.models[label] = spacy.load(model_path)
            else:
                print(f"Warning: Could not find trained model at {model_path}")
    
    def predict(self, text: str) -> List[Dict]:
        """
        Make predictions using all context models.
        
        Args:
            text: Input text
            
        Returns:
            List of predicted entities
        """
        all_entities = []
        
        # Run prediction for each model
        # If self.models is empty due to initialization failure, this safely returns an empty list.
        for label, model in self.models.items():
            doc = model(text)
            for ent in doc.ents:
                all_entities.append({
                    'text': ent.text,
                    'start_offset': ent.start_char,
                    'end_offset': ent.end_char,
                    'label': label
                })
        
        return all_entities
    
    def predict_all(self, data: List[Dict]) -> List[Dict]:
        """
        Make predictions for all examples in the data.
        
        Args:
            data: List of input examples
            
        Returns:
            List of examples with predictions
        """
        results = []
        for example in tqdm(data):
            text = example['text']
            predictions = self.predict(text)
            
            results.append({
                'id': example['id'],
                'text': text,
                'entities': predictions
            })
        
        return results
    
    def mark_entity_with_tags(self, text: str, entity_id: int, entities: list) -> tuple:
        """
        Enclose a specific entity with [NUM][/NUM] tags and adjust the offsets of other entities.
        
        Args:
            text: The original text
            entity_id: The ID of the entity to be marked
            entities: List of all entities with their offsets
            
        Returns:
            tuple: (marked_text, adjusted_entities)
                - marked_text: Text with the specified entity marked with [NUM][/NUM]
                - adjusted_entities: Entities with adjusted offsets
        """
        # Find the target entity
        target_entity = None
        for entity in entities:
            if entity["id"] == entity_id:
                target_entity = entity
                break
        
        if target_entity is None:
            print("Target entity is not found.")
            return text, entities.copy()  # Entity not found, return original
        
        # Extract start and end offsets of the target entity
        start_offset = target_entity["start_offset"]
        end_offset = target_entity["end_offset"]
        
        # Split the text into three parts: before entity, entity, after entity
        before_entity = text[:start_offset]
        entity_text = text[start_offset:end_offset]
        after_entity = text[end_offset:]
        
        before_ws = ""
        if len(before_entity) > 0 and before_entity[-1] != ' ' :
            before_ws = " "
        
        after_ws = ""
        if len(after_entity) > 0 and after_entity[0] != ' ' :
            after_ws = ""
        
        # Create the marked text
        marked_text = before_entity + before_ws + self.NUM_TOKEN_START + entity_text + self.NUM_TOKEN_END + after_ws + after_entity
        
        # Calculate the shift in offsets
        pre_tag_len = len(self.NUM_TOKEN_START) + len(before_ws)
        post_tag_len = len(self.NUM_TOKEN_END) + len(after_ws)
        total_shift = pre_tag_len + post_tag_len
        
        # Adjust offsets for all entities
        adjusted_entities = []
        for entity in entities:
            adjusted_entity = entity.copy()
            
            if entity["id"] == entity_id:
                adjusted_entity["start_offset"] += pre_tag_len
                adjusted_entity["end_offset"] += pre_tag_len
            # For entities that start after the target entity
            elif entity["start_offset"] >= end_offset:
                adjusted_entity["start_offset"] += total_shift
                adjusted_entity["end_offset"] += total_shift
            # For entities that end after the target entity starts but begin before it ends
            # (overlapping with the end of the target entity)
            elif entity["end_offset"] > start_offset and entity["start_offset"] < end_offset:
                
                print("overlapping span!")
                
                # This is a complex case of overlapping entities
                # Adjust based on where the overlap occurs
                if entity["start_offset"] < start_offset:
                    # Starts before target, ends inside or after target
                    if entity["end_offset"] <= end_offset:
                        # Ends inside target - extend end by pre_tag_len
                        adjusted_entity["end_offset"] += pre_tag_len
                    else:
                        # Ends after target - extend end by total_shift
                        adjusted_entity["end_offset"] += total_shift
                else:
                    # Starts inside target
                    adjusted_entity["start_offset"] += pre_tag_len
                    if entity["end_offset"] <= end_offset:
                        # Contained within target
                        adjusted_entity["end_offset"] += pre_tag_len
                    else:
                        # Ends after target
                        adjusted_entity["end_offset"] += total_shift
            
            # For entities that end at or before the target entity starts
            # No change needed for entities that end before the target starts
            
            adjusted_entities.append(adjusted_entity)
        
        return marked_text, adjusted_entities
    
    def convert_data(self, data: List[Dict]) -> List[Dict]:
        """
        Convert data into a format suitable for context extraction.
        
        Args:
            data: Data to be converted
            
        """
        
        converted_data = []
        id_counter = 1  # Initialize ID counter
        
        for example in data :
            text = example['text']
            entities = example['entities']
            relations = example['relations']
            
            id_to_ent = {entities[i]["id"]:i for i in range(len(entities))}
            
            for ent in entities :
                if ent['label'] not in self.labels :
                    related_ents = []
                    related_rels = []
                    # Identify the relevant context for numerical entities
                    for rel in relations :
                        if rel['from_id'] == ent['id'] :
                            related_ents.append(entities[id_to_ent[rel['to_id']]])
                            related_rels.append(rel)
                        elif rel['to_id'] == ent['id'] :
                            related_ents.append(entities[id_to_ent[rel['from_id']]])
                            related_rels.append(rel)
                    
                    related_ents.append(ent)
                    marked_text, related_ents = self.mark_entity_with_tags(text, ent['id'], related_ents)
                    
                    converted_data.append({
                        "id": id_counter,  # Use sequential ID
                        "text": marked_text,
                        "entities": related_ents,
                        "relations": related_rels
                    })
                    id_counter += 1  # Increment counter
        
        return converted_data
    
    def adjust_offsets_after_removing_tags(self, original_text: str, tagged_text: str, entities: list) -> list:
        """
        Adjust entity offsets after removing [NUM] tags from text.
        
        Args:
            original_text: Original text without tags
            tagged_text: Text with [NUM] and [/NUM] tags
            entities: List of entities with offsets based on tagged text
            
        Returns:
            list: Entities with adjusted offsets for the original text
        """
        # Define possible tag combinations
        possible_start_tags = ["[NUM] ", " [NUM] "]
        possible_end_tags = [" [/NUM]", " [/NUM] "]
        
        # Find the correct tag combination by trying each possibility
        start_tag = None
        end_tag = None
        
        for s_tag in possible_start_tags:
            for e_tag in possible_end_tags:
                # Create a temporary text with tags removed
                temp_text = tagged_text
                while s_tag in temp_text and e_tag in temp_text:
                    start_idx = temp_text.find(s_tag)
                    if start_idx == -1:
                        break
                        
                    end_idx = temp_text.find(e_tag, start_idx)
                    if end_idx == -1:
                        break
                    
                    # Remove the tags
                    before = temp_text[:start_idx]
                    content = temp_text[start_idx + len(s_tag):end_idx]
                    after = temp_text[end_idx + len(e_tag):]
                    temp_text = before + content + after
                
                # Check if cleaned text matches original
                if temp_text == original_text:
                    start_tag = s_tag
                    end_tag = e_tag
                    break
                    
            if start_tag and end_tag:
                break
        
        # If no matching tag combination found, return original entities
        if not start_tag or not end_tag:
            print("Warning: Could not determine the correct tag combination.")
            return entities.copy()
            
        # Identify all tag positions in the tagged text
        tag_positions = []
        i = 0
        
        while i < len(tagged_text):
            # Check for start tag
            if tagged_text[i:i+len(start_tag)] == start_tag:
                tag_positions.append({"pos": i, "tag_len": len(start_tag), "is_start": True})
                i += len(start_tag)
            # Check for end tag
            elif tagged_text[i:i+len(end_tag)] == end_tag:
                tag_positions.append({"pos": i, "tag_len": len(end_tag), "is_start": False})
                i += len(end_tag)
            else:
                i += 1
        
        # Sort tag positions by position
        tag_positions.sort(key=lambda x: x["pos"])
        
        # Adjust entity offsets
        adjusted_entities = []
        
        for entity in entities:
            adjusted_entity = entity.copy()
            start_pos = entity["start_offset"]
            end_pos = entity["end_offset"]
            
            # Calculate offset adjustments for each entity
            start_shift = 0
            end_shift = 0
            
            for tag in tag_positions:
                tag_pos = tag["pos"]
                tag_len = tag["tag_len"]
                
                # Tags before entity start affect start position
                if tag_pos < start_pos:
                    start_shift -= tag_len
                
                # Tags before entity end affect end position
                if tag_pos < end_pos:
                    end_shift -= tag_len
            
            # Apply the shifts
            adjusted_entity["start_offset"] = start_pos + start_shift
            adjusted_entity["end_offset"] = end_pos + end_shift
            
            adjusted_entities.append(adjusted_entity)
        
        return adjusted_entities