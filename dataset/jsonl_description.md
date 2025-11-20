## dataset_with_metadata.jsonl

JSON Lines file (UTF-8), one record per line. Keys are ordered with `id` first.

### Fields
- **id** (string): Unique identifier for this row; stable across re-runs; never shared across rows.
- **parent_document_id** (string): Identifier of the parent document that this row comes from.
- **dataset_name** (string, optional): Name of the source dataset.
- **url** (array of string, optional): URLs related to the parent document (e.g., arXiv, openreview, PWC, Figshare). Consolidated from multiple URL fields.
- **paper_title** (string, optional): Title of the paper or parent document.
- **categories** (string, optional): Figshare category path or label string provided by Figshare (format may vary by source).
- **modalities** (array of string, optional): Modalities from Papers with Code (e.g., image, video, text).
- **title** (string, optional): Generic title field used by some sources; when both exist, `paper_title` is the preferred paper-specific title.
- **parent_category_id** (integer, optional): Top-most parent category ID on Figshare (root category's parent id).
- **text** (string): Raw text span used for annotation and numeric extraction.
- **entities** (array of object): Entity annotations found in `text`.
  - **id** (integer): Entity identifier within the row.
  - **label** (string): Entity type label (e.g., DATA_COUNT).
  - **start_offset** (integer): Start character offset in `text` (inclusive).
  - **end_offset** (integer): End character offset in `text` (exclusive).
- **relations** (array of object): Relations between entities.
  - **id** (integer): Relation identifier within the row.
  - **from_id** (integer): Source entity `id`.
  - **to_id** (integer): Target entity `id`.
  - **type** (string): Relation type label.
