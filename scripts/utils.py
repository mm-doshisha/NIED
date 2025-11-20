import json
from typing import List, Dict, Any, Tuple, Optional

from scripts.combined_extractor import CombinedExtractor
from scripts.data_processor import DataProcessor

def load_data(file_path, all_labels_config="labels/all_labels.txt", numeric_labels_config="labels/numeric_labels.txt") :
    processor = DataProcessor(all_labels_config, numeric_labels_config)

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Process documents
    processed_docs = processor.process_documents(data)

    return processed_docs

def to_combined_entities(entities, relations) :
    numeric_entities = []
    context_entities = []
    for ent in entities :
        if ent['is_value'] :
            ent['contexts'] = []
            numeric_entities.append(ent)
        else :
            context_entities.append(ent)
    
    
    num_ent_to_id = {numeric_entities[idx]['id']:idx for idx in range(len(numeric_entities))}
    ctx_ent_to_id = {context_entities[idx]['id']:idx for idx in range(len(context_entities))}
    
    for rel in relations :
        
        if rel['from_id'] in num_ent_to_id.keys() :
            num_id = num_ent_to_id[rel['from_id']]
            ctx_id = ctx_ent_to_id[rel['to_id']]
        else :
            num_id = num_ent_to_id[rel['to_id']]
            ctx_id = ctx_ent_to_id[rel['from_id']]
        
        numeric_entities[num_id]['contexts'].append(context_entities[ctx_id])
    
    return numeric_entities


def save_data(file_path: str, data: List[Dict]) -> None:
    """
    Save a list of dictionaries to a file in JSONL format.
    
    Args:
        file_path: Path where the JSONL file will be saved
        data: List of dictionaries to be saved
    
    Returns:
        None
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Successfully saved {len(data)} items to {file_path}")
