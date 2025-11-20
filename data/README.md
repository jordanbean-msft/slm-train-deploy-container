# Training Data Directory

## Purpose

This directory is where you place your training data for fine-tuning the SLM.

## Required File

Place your training data at: `data/training_data.jsonl`

## Format

Each line must be valid JSON with the following structure:

```jsonl
{"prompt": "Your input text here", "completion": "Expected output text here"}
{"prompt": "Another input", "completion": "Another output"}
```

## Example

```jsonl
{"prompt": "Translate to French: Hello", "completion": "Bonjour"}
{"prompt": "Summarize: The quick brown fox jumps over the lazy dog.", "completion": "A fox jumps over a dog."}
{"prompt": "What is the capital of France?", "completion": "The capital of France is Paris."}
```

## Requirements

- Each line must be valid JSON
- Required fields: `prompt` and `completion`
- Both fields must be non-empty strings
- Maximum prompt length: 2048 tokens (configurable in `configs/data_config.yaml`)
- Maximum completion length: 1024 tokens (configurable in `configs/data_config.yaml`)
- Minimum samples: 10 (configurable)

## Workflow

1. **Place your data**: Create `data/training_data.jsonl` with your training examples
2. **Run validation**: Execute `notebooks/02-prepare-data.ipynb` to validate format
3. **Upload**: The notebook will split, upload to Azure Blob Storage, and register as Azure ML dataset
4. **Train**: Use the registered dataset in training jobs via `notebooks/05-train-model.ipynb`

## Processed Data

After running the preparation notebook, processed files will be created in:

- `data/processed/train.jsonl` - Training set (80% by default)
- `data/processed/val.jsonl` - Validation set (20% by default)
- `data/processed/manifest.json` - Metadata about the split

## Configuration

Edit `configs/data_config.yaml` to change:

- Split ratio (default: 80/20)
- Token length limits
- Azure blob storage prefix
- Dataset name and version
