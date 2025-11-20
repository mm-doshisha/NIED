import json
import re
from typing import Dict, List, Any, Optional, Set


class DataProcessor:
    def __init__(self, all_labels_path: str, numeric_labels_path: str):
        """
        Initialize DataProcessor with paths to label definition files.
        
        Args:
            all_labels_path: Path to the all labels text file
            numeric_labels_path: Path to the numeric labels text file
        """
        # Load label definitions
        self.all_labels = self._load_labels(all_labels_path)
        self.numeric_labels = self._load_labels(numeric_labels_path)
        
        # Create sets for quick checking
        self.valid_labels = set(self.all_labels)
        self.value_labels = set(self.numeric_labels)
    
    def _load_labels(self, file_path: str) -> List[str]:
        """Load labels from text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _simple_tokenize(self, text: str) -> List[Dict]:
        """
        Simple tokenizer that splits text by word boundaries.
        Returns tokens with their offsets.
        """
        tokens = []
        token_id = 0
        # Simple word boundary pattern
        pattern = r'\b\w+\b|[^\w\s]'
        
        for match in re.finditer(pattern, text):
            token_text = match.group(0)
            start = match.start()
            end = match.end()
            
            tokens.append({
                "token_id": token_id,
                "text": token_text,
                "start": start,
                "end": end,
                "entities": [{"label": "O", "entity_id": None}]
            })
            token_id += 1
        
        return tokens
    
    def _assign_entities_to_tokens(self, tokens: List[Dict], entities: List[Dict]) -> List[Dict]:
        """
        Assign entity labels to tokens.
        """
        # Create a mapping from character offsets to token indices
        char_to_token = {}
        for i, token in enumerate(tokens):
            for pos in range(token["start"], token["end"]):
                char_to_token[pos] = i
        
        # For each entity, find tokens it spans and assign labels
        for entity in entities:
            start_offset = entity["start_offset"]
            end_offset = entity["end_offset"]
            label = entity["label"]
            entity_id = entity["id"]
            
            # Find token indices that overlap with this entity
            token_indices = set()
            for pos in range(start_offset, end_offset):
                if pos in char_to_token:
                    token_indices.add(char_to_token[pos])
            
            # Assign entity to these tokens
            for idx in token_indices:
                # If first entity or if "O" label
                if tokens[idx]["entities"][0]["label"] == "O":
                    tokens[idx]["entities"] = [{"label": label, "entity_id": entity_id}]
                else:
                    # Add to existing entities for this token
                    tokens[idx]["entities"].append({"label": label, "entity_id": entity_id})
        
        return tokens
    
    def process_document(self, doc: Dict) -> Optional[Dict]:
        """
        Process a single document according to the requirements.
        
        Args:
            doc: A document dictionary with text, entities, and relations
            
        Returns:
            Processed document or None if document should be discarded
        """
        # 必須フィールドのチェック
        if "text" not in doc:
            print("Warning: Document missing 'text' field, skipping...")
            return None
            
        # idが存在しない場合は自動生成
        doc_id = doc.get("id", f"doc_{hash(doc['text'])}")
        text = doc["text"]
        entities = doc.get("entities", [])
        relations = doc.get("relations", [])
        
        # Create a deep copy to avoid modifying the original
        processed_entities = []
        processed_relations = []
        invalid_relations = []
        
        # Step 1: Process entities (filter, add is_value attribute)
        entity_map = {}  # Map entity IDs to processed entities
        for entity in entities:
            entity_id = entity["id"]
            label = entity["label"]
            
            # Check if this label is valid
            if label not in self.valid_labels:
                continue
            
            # Create processed entity
            processed_entity = {
                "id": entity_id,
                "label": label,
                "start_offset": entity["start_offset"],
                "end_offset": entity["end_offset"],
                "is_value": label in self.value_labels
            }
            
            processed_entities.append(processed_entity)
            entity_map[entity_id] = processed_entity
        
        # Step 2: Validate relations and check for conflicts
        entity_has_relation = set()
        
        for relation in relations:
            relation_id = relation["id"]
            from_id = relation["from_id"]
            to_id = relation["to_id"]
            relation_type = relation["type"]
            
            # Check if both entities exist after processing
            if from_id not in entity_map or to_id not in entity_map:
                continue
            
            # Check head and tail label constraints
            head_entity = entity_map[from_id]
            tail_entity = entity_map[to_id]
            
            # Add to processed relations
            processed_relations.append({
                "id": relation_id,
                "from_id": from_id,
                "to_id": to_id,
                "type": relation_type
            })
            
            # Mark both entities as having relations
            entity_has_relation.add(from_id)
            entity_has_relation.add(to_id)
        
        # Step 3: Remove non-value entities without relations
        final_entities = []
        for entity in processed_entities:
            if entity["is_value"] or entity["id"] in entity_has_relation:
                # Remove is_value attribute from final output
                entity_copy = entity.copy()
                del entity_copy["is_value"]
                final_entities.append(entity_copy)
        
        # Step 4: Tokenize text and assign entities
        tokens = self._simple_tokenize(text)
        tokenized_text = self._assign_entities_to_tokens(tokens, final_entities)
        
        # Create final document
        processed_doc = {
            "id": doc_id,
            "text": text,
            # "tokenized_text": tokenized_text,
            "entities": final_entities,
            "relations": processed_relations
        }
        
        return processed_doc
    
    def process_documents(self, docs: List[Dict]) -> List[Dict]:
        """
        Process multiple documents.
        
        Args:
            docs: List of document dictionaries
            
        Returns:
            List of processed documents
        """
        processed_docs = []
        for doc in docs:
            processed_doc = self.process_document(doc)
            if processed_doc:
                processed_docs.append(processed_doc)
        
        return processed_docs
