from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import os # osモジュールを追加

from scripts.context_extractor import ContextExtractor
from scripts.numeric_extractor import NumericExtractor

class CombinedExtractor :
    
    def __init__(self, numeric_model_path=None, context_model_path=None) :
        
        if numeric_model_path :
            # NumericExtractor expects the full path including model-best, which is now ensured by main.py
            self.numeric_extractor = NumericExtractor(model=numeric_model_path)
            
        if context_model_path :
            # Context model path correction: Ensure path points to the directory containing model_* sub-dirs
            if context_model_path.endswith("/model-best"):
                # If path ends in model-best, trim it back to the base directory (e.g., models/context_model)
                context_model_path = context_model_path[:-11] 
            
            # ContextExtractor handles loading individual model-best sub-directories internally
            self.context_extractor = ContextExtractor(model=context_model_path)
        
            
    
    def predict(self, text, text_id=None) :
        if not self.numeric_extractor :
            print("Numeric extractor is unvalid.")
            return
        if not self.context_extractor :
            print("Context extractor is unvalid.")
            return
        
        num_entities = self.numeric_extractor.predict(text, text_id)

        relations = []
        entities = []
        for n_ent in num_entities :
            tagged_text, adjust_entities = self.context_extractor.mark_entity_with_tags(text, n_ent['id'], [n_ent])
            context_entities = self.context_extractor.predict(tagged_text)
            # FIX: The adjust_offsets_after_removing_tags method is now correctly guaranteed to exist in ContextExtractor
            context_entities = self.context_extractor.adjust_offsets_after_removing_tags(text, tagged_text, context_entities)

            for idx in range(len(context_entities)) :
                c_ent = context_entities[idx]
                c_ent['id'] = int(str(n_ent['id']) + "000" + str(idx + 1))
                entities.append({
                    "id": c_ent['id'],
                    "text": text[c_ent['start_offset']:c_ent['end_offset']],
                    "start_offset": c_ent['start_offset'],
                    "end_offset": c_ent['end_offset'],
                    "label": c_ent['label'],
                    "is_value": False
                })
                relations.append({
                    "type": "HAS_" + c_ent['label'],
                    "from_id": n_ent['id'],
                    "to_id": c_ent['id'],
                })
                
            entities.append({
                "id": n_ent['id'],
                "text": text[n_ent['start_offset']:n_ent['end_offset']],
                "start_offset": n_ent['start_offset'],
                "end_offset": n_ent['end_offset'],
                "label": n_ent['label'],
                "is_value": True
            })
        
        return entities, relations
    
    def predict_all(self, data:List[Dict]) :
        if not self.numeric_extractor :
            print("Numeric extractor is unvalid.")
            return
        if not self.context_extractor :
            print("Context extractor is unvalid.")
            return

        results = []
        for example in tqdm(data) :
            results.append(example)
            ents, rels = self.predict(example['text'])
            results[-1]['entities'] = ents
            results[-1]['relations'] = rels
        
        return results