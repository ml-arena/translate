# English-German Translation Environment

A predict-only supervised learning environment for evaluating pre-trained translation models.

## Overview

This environment tests agents on English-to-German translation tasks WITHOUT providing training data. Agents must use pre-trained models (e.g., from Hugging Face transformers) to translate English sentences into German.

**Key Features:**
- **Predict-only workflow:** No training data provided during competition
- **Public dataset:** 98 common English-German sentence pairs for testing
- **Simple evaluation:** Token overlap metric (F1 score)
- **Production-ready:** Replace with private dataset for ML-Arena competitions

## Installation

```bash
# From the translate directory
pip install -e .

# Or build wheel
python setup.py bdist_wheel
pip install dist/translate-*.whl
```

## Dataset Preparation

Generate the public English-German dataset:

```bash
python tools/prepare_data.py
```

This creates `translate/data/en_de_dataset.json` with 98 sentence pairs.

**For production:** Replace `en_de_dataset.json` with your private dataset in the same format.

## Environment API

### Initialization

```python
from translate import Env

env = Env(number_episodes=20)  # 20 translation tasks
```

### Methods

**`get_next_task()` → dict**
```python
task = env.get_next_task()
# Returns:
{
    'X_test': np.array(['Hello', 'Good morning', ...]),  # English sentences
    'y_test': np.array(['Hallo', 'Guten Morgen', ...])   # German references (for eval only)
}
```

**`evaluate(predictions, true_labels)` → float**
```python
score = env.evaluate(predictions, references)
# Returns: F1 score between 0 and 1 (token overlap metric)
```

**`reset()`**
```python
env.reset()  # Reset to start from first task
```

**`is_complete()` → bool**
```python
if env.is_complete():
    print("All tasks completed")
```

## Agent Example

Agents receive English sentences and must return German translations:

```python
class TranslationAgent:
    def __init__(self):
        # Load pre-trained model (e.g., from Hugging Face)
        from transformers import pipeline
        self.translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")

    def reset(self):
        pass  # No state to reset

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Translate English sentences to German"""
        translations = []
        for sentence in X_test:
            result = self.translator(sentence)[0]['translation_text']
            translations.append(result)
        return np.array(translations, dtype=object)
```

## Dataset Format

The dataset JSON file has this structure:

```json
{
    "description": "English-German translation dataset",
    "num_pairs": 98,
    "pairs": [
        {"en": "Hello, how are you?", "de": "Hallo, wie geht es dir?"},
        {"en": "Good morning.", "de": "Guten Morgen."},
        ...
    ]
}
```

## Competition Workflow

1. **Agent loads pre-trained model** (e.g., `Helsinki-NLP/opus-mt-en-de`)
2. **Environment provides English sentences** via `get_next_task()`
3. **Agent translates** sentences via `predict(X_test)`
4. **Environment evaluates** quality via `evaluate(predictions, references)`
5. **Repeat** for all tasks (typically 20 tasks × 10 sentences = 200 total)

## Evaluation Metric

The environment uses a simplified token-overlap F1 score:

```
Precision = |predicted_tokens ∩ reference_tokens| / |predicted_tokens|
Recall    = |predicted_tokens ∩ reference_tokens| / |reference_tokens|
F1        = 2 * (Precision * Recall) / (Precision + Recall)
```

**For production:** Replace with proper MT metrics (BLEU, chrF, COMET, etc.)

## Versioning

Current version: **0.1**

To increment version:
```bash
# From envs_development directory
make tag-version ENV_NAME=translate
```

## ML-Arena Integration

### Local Testing

Copy to storage for local testing:
```bash
# From envs_development directory
make dev-copy ENV_NAME=translate
```

### Docker Image

The environment is installed in the base Docker image:
```dockerfile
# imagemanager/custom_images/base/Dockerfile.jinja
RUN pip install git+https://github.com/ml-arena/translate.git@translate_v0.1
```

## Differences from PermutedMNIST

| Aspect | PermutedMNIST | Translate |
|--------|---------------|-----------|
| **Workflow** | Train + Predict | Predict-only |
| **Training data** | Provided (X_train, y_train) | NOT provided |
| **Agent type** | Trains during competition | Uses pre-trained model |
| **Task type** | Meta-learning | Zero-shot evaluation |
| **Data type** | Numpy arrays (images) | Text (strings) |
| **Metric** | Classification accuracy | Translation F1 (token overlap) |

## Notes

- **No training allowed:** Agents cannot call any `train()` method during competition
- **Pre-trained models:** Agents must load models before competition starts
- **Public dataset:** Current dataset is public for testing; use private for competitions
- **Simple metric:** Token overlap is simple; consider BLEU/chrF for production

## License

MIT License - Public domain dataset for testing purposes.