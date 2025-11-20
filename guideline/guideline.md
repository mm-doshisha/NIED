# NIED: Annotation Guidelines for Numerical Information Extraction from Dataset Descriptions

## Overview

NIED (Numerical Information Extraction from Dataset descriptions) is an annotated corpus comprising 3,926 dataset descriptions from academic papers and repositories. The corpus employs a two-tier labeling scheme that separates numerical entities (corresponding to quantitative aspects) from non-numerical entities (providing semantic context) to enable structured metadata extraction from unstructured dataset documentation.

## Labeling Scheme

### Numerical Labels

### DATA_COUNT

- **Definition**: General counts related to data quantity, including overall datasets, train/validation/test splits, pre-processing data counts, counts per specific target, and subset counts.
- **Examples**:
    - Dataset totals: "The dataset contains a total of **15,000** images"
    - Training data: "The training set contains **50,000** examples"
    - Validation data: "**2,000** samples were used for validation"
    - Test data: "The test set contains **10,000** examples"
    - Original data: "The original corpus contained **2 million** sentences before filtering"
    - Per-target counts: "Each class contains approximately **1,000** samples"
    - Subset counts: "The dataset is divided into **5** distinct subsets"

### ANNOTATION

- **Definition**: Counts of annotation types assigned to data, such as classes, categories, tags, and features.
- **Examples**:
    - "Images are labeled with one of **20** categories"
    - "The text is annotated with **35** different POS tags"
    - "Each image contains up to **10** bounding boxes"
    - "We track **17** body joints in each frame"

### COLLECTION

- **Definition**: Integrated label for counts related to data collection, including number of subjects/objects/events, attributes of collection targets, collection conditions, and data sources.
- **Examples**:
    - Collection subjects: "Speech from **200** different speakers"
    - Collection objects: "We photographed **300** different buildings"
    - Subject attributes: "Speakers from **20** different dialects"
    - Collection conditions: "Recorded in **12** different sessions"
    - Geographic conditions: "Data collected at **5** geographic locations"

### DOMAIN

- **Definition**: Number of target domains for data, such as languages, fields, or themes.
- **Examples**:
    - "The corpus includes text in **10** languages"
    - "Images from **5** medical specialties"
    - "Text from **8** distinct genres"

### CONTRIBUTOR

- **Definition**: Number of people involved in dataset creation, including annotators, evaluators, and validators.
- **Examples**:
    - "Annotated by **5** experts"
    - "Verified by **3** independent annotators"
    - "Ratings from **50** crowdworkers"

### Non-numerical Labels

### UNIT

- **Definition**: Basic units or standards for measuring numerical values, typically **single words** that follow numbers directly. For consecutive nouns where one does not modify the other and removing one would completely change the meaning, spans of 2+ words are permitted.
- **Examples**:
    - "We analyzed 24 **frames** from the video dataset"
    - "We collected 1000 Japanese **sentences** for analysis"
    - "The corpus includes 5 **documents** with varying lengths"
    - "Our dataset contains 1000 **images** of sleeping cats"
    - "The training set includes 15,000 **examples** for machine learning"

### EXTENDED_UNIT

- **Definition**: Complete noun phrases serving as units, including pre-modifiers and "of" phrases only. Post-modifying prepositional phrases (except "of") are excluded.
- **Examples**:
    - "We analyzed 1000 **color images of sleeping cats** for the study"
    - "The model was trained on 50,000 **training examples** from various sources"
    - "We collected 200 **Japanese text samples** for linguistic analysis"
    - "The satellite dataset contains 3000 **high-resolution satellite images** for mapping"

### QUALIFIER

- **Definition**: Supplementary or modifying information about numbers, including characteristics, sources, purposes, processing methods, and split information.
- **Examples**:
    - Source: "We used 15 cell lines **derived from human lung tissue** for the experiment"
    - Collection method/source: "The corpus includes 30,000 texts **collected from social media platforms**"
    - Specific examples/purpose: "The annotation uses 4 categories: **PER, LOC, ORG, MISC** for entity classification"
    - Content/purpose: "We analyzed 100 images **containing cats, dogs and fish** for object detection"
    - Purpose/objective: "We collected 1,000 samples **for training purposes**"
    - Processing method: "The dataset contains 5,000 examples **processed with automated tools**"
    - Conditional notation: "We analyzed 100 documents **(200 if duplicates are included)**"

### MODIFIER

- **Definition**: Expressions indicating accuracy, range, or approximation of numerical values, typically appearing before numbers.
- **Examples**:
    - "The dataset contains **approximately** 1000 samples from various sources"
    - "Each image has **at least** 3 annotations for quality control"
    - "The video sequences contain **up to** 50 frames per second"
    - "We recruited **more than** 200 participants for the study"
    - "The system processes **an average of** 3 images per batch"
- **Note**: Limiting words (only, just, merely), ordinal words (first, initial, last), and ranking words (top, best, worst) do not indicate numerical range/approximation and are excluded from MODIFIER

### DENOMINATOR

- **Definition**: Complete noun phrases serving as denominators in ratios or densities, expressing "per" or "each" relationships. This includes the entire noun phrase (with articles, numbers, and modifiers) that contextually functions as the denominator, even when explicit "per"/"each" expressions are absent.
- **Examples**:
    - "We found 10 objects **per image** in the dataset"
    - "Each document has 3 annotations for **each sentence**"
    - "The model uses 5 samples **per class** for training"
    - "On average, **each table** contains 100 rows"
    - "**The 50,000 documents** each have between 2 and 5 pages"

### SPLIT

- **Definition**: Dataset split terms only (training, test, validation, dev, holdout), excluding accompanying words like "set" or "data".
- **Examples**:
    - "We used the **training** set with 50,000 samples for model development"
    - "The model was evaluated on a **validation** split containing 10,000 examples"
    - "Performance was measured on the **test** portion with 5,000 instances"
    - "We set aside **development** data consisting of 2,000 samples for hyperparameter tuning"
    - "The **dev** set contains 1,000 examples for validation"
    - "All **train** data comprising 80,000 instances was preprocessed before training"

### DATASET

- **Definition**: Data sources and repositories that store, provide, or manage data, including datasets, corpora, databases, repositories, research papers/literature (as data sources), benchmarks, archives, and collections. Articles and modifying phrases are included.
- **Examples**:
    - "We collected 3,000 images from **the ImageNet dataset** for our experiments"
    - "**The medical corpus** contains 50,000 patient records from multiple hospitals"
    - "Our analysis used 25,000 articles extracted from **the Wikipedia database**"
    - "**This research collection** includes 1,200 documents spanning 10 different domains"

## Span Definition and Annotation Principles

### What is a Span?

A span refers to the range of text that serves as an annotation target. In numerical information extraction tasks, there are two types: numerical spans and non-numerical spans.

### Basic Principles

### Boundary Determination Principles

- **Semantic Completeness**: Minimum unit that is semantically complete
- **Sentence Constraint**: No annotation across sentence boundaries
- **Overlap Permission**: Multiple labels can be assigned to the same span

### Span Types

### Numerical Spans

Applied to actual numerical values (digits, symbols, range expressions).

- **Examples**:
    - "**1,000**"
    - "**50**"
    - "**between 100 and 200**" (entire range expression)
    - "**from 2010 to 2020**" (entire range expression)

Range expressions ("from A to B", "between X and Y") are treated as single spans. Numbers that are part of proper nouns (SODA10M, CIFAR-10) are not labeled.

### Non-numerical Spans

Applied to text other than numbers (words, phrases, noun phrases).

- **Examples**:
    - "**training**" (SPLIT)
    - "**images**" (UNIT)
    - "**approximately**" (MODIFIER)
    - "**from social media**" (QUALIFIER)

## Boundary Determination Rules

### 1. Article and Possessive Inclusion Principle

Non-numerical spans generally include articles (the, a, an) and possessives (our, their, etc.).

- **Examples**:
    - "**the training set**" (DATASET)
    - "**our dataset**" (DATASET)
    - "**an average of**" (MODIFIER)

### 2. Parentheses Inclusion Principle

For non-numerical spans other than EXTENDED_UNIT, UNIT, MODIFIER, and DENOMINATOR, parentheses are included only when they supplement or modify the non-numerical span.

- **Examples**:
    - "**The Common Voice corpus (CV-11.0)**" (DATASET)
    - "**emotion categories**" (EXTENDED_UNIT - parentheses excluded)

### 3. Modifier Separation Principle

When multiple modifiers (MODIFIER, QUALIFIER) exist, they are processed as separate spans.

- **Example**: "approximately more than 500"
    - "**approximately**" → MODIFIER
    - "**more than**" → MODIFIER

### 4. QUALIFIER Division Judgment Method

### Basic Rule

Different types of modifying content are divided into separate QUALIFIER spans.

### Division Judgment Steps

**Step 1**: Look for conjunctions and relative pronouns

- Check modifying content connected by "and", ",", "which", "that", etc.

**Step 2**: Identify types of modifying content

- **Source/Collection location**: "from X", "collected from Y"
- **Conditions/Environment/Methods**: "in X condition", "under Y", "using Z"
- **Results/Effects/Characteristics**: "which causes X", "that improves Y"

**Step 3**: Make division judgment

- Different types of modification → Divide (multiple QUALIFIERs)
- Same type of modification → Integrate (single QUALIFIER)

## Complex Expression Processing

### Range Expressions

Entire range expressions are treated as single numerical labels.

- **Target patterns**:
    - "between A and B"
    - "from A to B"
    - "A to B"
    - "A~B"
    - "A-B" (hyphen-connected)

### Parenthetical Information

Numbers and units within parentheses are also labeled.

- **Example**: "We recruited 400 participants (200 children and 200 adults)"
    - "**400**" → COLLECTION
    - "**participants**" → UNIT
    - "**200**" (first occurrence) → COLLECTION
    - "**children**" → UNIT
    - "**200**" (second occurrence) → COLLECTION
    - "**adults**" → UNIT
    - “**(200 children and 200 adults)**” → QUALIFIER

### List Format

Each item receives appropriate individual labels.

- **Example**: "1,000 images, 500 videos, and 200 audio files"
    - Each number and unit is annotated individually

## Processing Elliptical Expressions

### Complementing Omitted Units

Units are complemented and annotated only when they exist within the same sentence.

- **Example**: "We used 1,000 images for training and 200 for testing"
    - "**1,000**" → DATA_COUNT
    - "**images**" → UNIT
    - "**200**" → DATA_COUNT
    - Complemented "images" (as unit for 200) → UNIT

### Pronoun Processing

- **Excluded**: Simple pronouns like "These", "They", "It"
- **Included**: Expressions with modifiers like "This dataset", "These images"

## Overlapping Label Examples

Multiple labels can be assigned to the same span. Overlaps of related concepts are also permitted.

**Example 1**: "50 training images"

- "50" → DATA_COUNT
- "training" → SPLIT
- "images" → UNIT
- "training images" → EXTENDED_UNIT

**Example 2**: "the training set contains 1000 examples"

- "the training set" → DATASET
- "training" → SPLIT
- "1000" → DATA_COUNT
- "examples" → UNIT

**Example 3**: "500 examples from the validation set"

- "500" → DATA_COUNT
- "examples" → UNIT
- "from the validation set" → QUALIFIER
- "validation" → SPLIT

## Corpus Statistics

| Statistic | Value |
| --- | --- |
| Total descriptions | 3,926 |
| Total sentences | 16,727 |
| Numerical entity spans | 2,000 |
| Non-numerical entity spans | 5,971 |

## Model Performance

The best performing model (XLNet) achieved:

- **Numerical entities**: 72.0% F1 score (exact matching)
- **Non-numerical entities**: 71.4% F1 score (exact matching)
- **Label-agnostic numerical detection**: 79.1% F1 score (exact matching)

These results demonstrate the feasibility of automated numerical information extraction from dataset documentation and support the development of tools for dataset discovery and comparison based on quantitative characteristics.