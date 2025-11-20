import json
import random
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import subprocess
import os
from scripts.data_processor import DataProcessor

class NumericExtractor:
    def __init__(self, model=None, output_path=None):
        """
        Initialize the NER model for extracting numerical entities.
        
        Args:
            model: Path to saved model (Used for test/predict)
            output_path: Directory to save the model (Used for train)
        """
        
        # Determine the target path for initialization/output.
        target_path = output_path if output_path is not None else model
        
        if target_path is None:
            # If neither model path nor output path is provided, stop execution.
            raise ValueError(
                "NumericExtractor requires either 'model' path (for prediction/testing) "
                "or 'output_path' (for training). None was provided."
            )

        self.output_path = Path(target_path)
        
        # Create output directory if it doesn't exist (needed for training/output saving)
        if output_path is not None:
             os.makedirs(self.output_path, exist_ok=True)

        # Initialize model
        self.nlp = None 
        
        # Load model if specified (for predict/test)
        if model and os.path.exists(model):
            try:
                self.nlp = spacy.load(model)
            except OSError as e:
                # If a model path is explicitly provided but fails to load, raise an error.
                raise OSError(f"Error loading model from path {model}. Cannot proceed without a trained model.") from e
        
        # If model is None (i.e., we are in the middle of training setup), initialize a blank pipe.
        if self.nlp is None:
            self.nlp = spacy.blank("en") # Start with a blank language object for training initialization
            
        # Create NER component if it doesn't exist
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner", last=True)
        else:
            self.ner = self.nlp.get_pipe("ner")
        
        # Initialize labels using DataProcessor
        self.processor = DataProcessor("labels/all_labels.txt", "labels/numeric_labels.txt")
        self.labels = list(self.processor.value_labels)
        
        # Add labels to the NER component
        for label in self.labels:
            self.ner.add_label(label)
        
        self.add_custom_tokenizer()
        
    def add_custom_tokenizer(self):
        return
            
    def prepare_training_data(self, train_data: List[Dict]):
        """
        Convert processed documents to spaCy training examples.
        Only include numerical/value entities.
        
        Args:
            train_data: List of training examples
        """
        
        doc_bin = DocBin()
        
        for training_example in tqdm(train_data): 
            text = training_example['text']
            labels = training_example['entities']
            doc = self.nlp.make_doc(text) 
            ents = []
            for label_example in labels:
                start = label_example['start_offset']
                end = label_example['end_offset']
                label = label_example['label']
                
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
                if span is None:
                    print("Skipping entity")
                else:
                    ents.append(span)
                    
            filtered_ents = filter_spans(ents)
            doc.ents = filtered_ents 
            doc_bin.add(doc)
        
        # Create spacy directory if it doesn't exist
        os.makedirs("spacy", exist_ok=True)
        doc_bin.to_disk("spacy/train.spacy") 
    
    def train(self, train_data: List[Dict], dev_examples: Optional[List[Dict]] = None, 
               config_path = "config/config_roberta-base.cfg", gpu_id = 0):
        """
        Train the NER model.
        
        Args:
            train_data: List of training examples
            dev_examples: List of development examples for evaluation
            config_path: Path of the finetuning config file
            gpu_id: GPU ID to use
        """
        
        self.prepare_training_data(train_data)
        subprocess.run("export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0", shell=True, check=True)
        
        # Ensure the main output directory exists before calling spacy train
        os.makedirs(self.output_path, exist_ok=True) 
        
        if dev_examples == None :
            subprocess.run(f"python -m spacy train {config_path} --output-path {self.output_path} --paths.train spacy/train.spacy --paths.dev spacy/train.spacy --gpu-id {gpu_id}", shell=True, check=True)

        # Load the trained model
        model_path = str(self.output_path) + "/model-best"
        if os.path.exists(model_path):
            self.nlp = spacy.load(model_path)
        else:
            print(f"Warning: Could not find trained model at {model_path}")
       
    def predict(self, text: str, text_id = None) -> List[Dict]:
        """
        Predict numerical entities in a text.
        
        Args:
            text: Input text
            
        Returns:
            List of predicted entities
        """
        doc = self.nlp(text)
        entities = []
        
        pre_id = (str(text_id)) if text_id else ""
        id = 1
        for ent in doc.ents:
            if ent.label_ in self.labels:
                entities.append({
                    "id": int(pre_id + "00" +str(id)),
                    "text": ent.text,
                    "label": ent.label_,
                    "start_offset": ent.start_char,
                    "end_offset": ent.end_char,
                })
                id += 1
                
        
        return entities