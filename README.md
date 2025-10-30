# English-German Translation Environment

A predict-only supervised learning environment for evaluating pre-trained translation models.

## Overview

This environment tests agents on English-to-German translation tasks WITHOUT providing training data. Agents must use pre-trained models (e.g., PyTorch models) to translate English sentences into German.

**Key Features:**
- **Predict-only workflow:** No training data provided during competition
- **Public dataset:** 98 common English-German sentence pairs for testing
- **BLEU evaluation:** Standard MT metric using sacrebleu library
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
# Returns: BLEU score between 0 and 100 (higher is better)
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

## Agent Examples

### Example 1: PyTorch Pre-trained Model

Agents receive English sentences and must return German translations:

```python
import torch
import numpy as np

class TranslationAgent:
    def __init__(self):
        # Load your pre-trained PyTorch translation model
        # This could be a model trained offline or downloaded
        self.model = torch.load('path/to/translation_model.pt')
        self.model.eval()

        # Or load from state dict
        # self.model = MyTranslationModel()
        # self.model.load_state_dict(torch.load('model_weights.pth'))

    def reset(self):
        pass  # No state to reset

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Translate English sentences to German"""
        translations = []

        with torch.no_grad():
            for sentence in X_test:
                # Tokenize and prepare input
                input_tensor = self.tokenize(sentence)

                # Generate translation
                output = self.model(input_tensor)
                translation = self.decode(output)

                translations.append(translation)

        return np.array(translations, dtype=object)

    def tokenize(self, sentence):
        # Implement your tokenization logic
        pass

    def decode(self, output):
        # Implement your decoding logic
        pass
```

### Example 2: Random Baseline (for testing)

A simple random baseline agent for testing the environment:

```python
import numpy as np
import random

class RandomAgent:
    """Random baseline agent that generates random German words"""

    def __init__(self):
        # Common German words for random baseline
        self.german_words = [
            "Hallo", "Guten", "Morgen", "Tag", "Abend", "Nacht",
            "danke", "bitte", "ja", "nein", "gut", "sehr",
            "ist", "der", "die", "das", "und", "oder",
            "ich", "du", "er", "sie", "wir", "ihr",
            "haben", "sein", "werden", "können", "müssen",
            "hier", "dort", "heute", "morgen", "gestern"
        ]

    def reset(self):
        pass  # No state to reset

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate random German 'translations'"""
        translations = []

        for sentence in X_test:
            # Generate random translation with 3-7 words
            num_words = random.randint(3, 7)
            translation = " ".join(random.choices(self.german_words, k=num_words))
            translations.append(translation)

        return np.array(translations, dtype=object)
```

**Expected BLEU scores:**
- Random baseline: ~0-5 BLEU
- Good model: 40-60 BLEU
- Excellent model: 60+ BLEU

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

1. **Agent loads pre-trained model** (PyTorch, TensorFlow, etc.)
2. **Environment provides English sentences** via `get_next_task()`
3. **Agent translates** sentences via `predict(X_test)`
4. **Environment evaluates** quality via `evaluate(predictions, references)`
5. **Repeat** for all tasks (typically 20 tasks × 10 sentences = 200 total)

## Evaluation Metric

The environment uses **BLEU (Bilingual Evaluation Understudy)** score via the `sacrebleu` library.

**What is BLEU?**
- Standard metric for machine translation quality
- Measures n-gram overlap between predictions and references
- Score range: 0-100 (higher is better)
- Industry standard used in MT research and competitions

**Implementation:**
```python
from sacrebleu import corpus_bleu

# sacrebleu handles tokenization, smoothing, and scoring
bleu = corpus_bleu(predictions, [references])
score = bleu.score  # 0-100
```

**Interpretation:**
- **60-100:** Excellent translation (near human quality)
- **40-60:** Good translation (understandable, mostly correct)
- **20-40:** Fair translation (partially understandable)
- **0-20:** Poor translation (barely recognizable)

**For advanced competitions:** Consider adding chrF, COMET, or TER metrics alongside BLEU

## Versioning

Current version: **0.2**

**Changelog:**
- v0.2: Replaced token overlap with BLEU metric (sacrebleu)
- v0.1: Initial release with token overlap metric

