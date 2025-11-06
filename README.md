# English-German Translation Environment

A predict-only environment for evaluating pre-trained translation models.

Competition: https://ml-arena.com/viewcompetition/14

## Overview

Translate English sentences to German using your pre-trained model. No training data is provided during the competition.

## Competition Submission Requirements

Your submission must include:
- **File:** `agent.py`
- **Class:** `Agent`
- **Method:** `def predict(self, X_test: np.ndarray) -> np.ndarray`

You can include additional methods, import files, or `.pt` weight files as needed.

## Agent Example

Your `agent.py` with a PyTorch model:

```python
import torch
import numpy as np

class Agent:
    def __init__(self):
        # Load your pre-trained translation model
        self.model = torch.load('translation_model.pt')
        self.model.eval()

        # Or load from separate files
        # from model_architecture import TranslationModel
        # self.model = TranslationModel()
        # self.model.load_state_dict(torch.load('weights.pt'))

    def reset(self):
        pass  # Optional: reset agent state

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Translate English sentences to German"""
        translations = []

        with torch.no_grad():
            for sentence in X_test:
                # Your translation logic here
                translation = self.translate(sentence)
                translations.append(translation)

        return np.array(translations, dtype=object)

    def translate(self, sentence):
        # Your model inference logic
        # tokenize -> model forward -> decode
        pass
```

## Evaluation

Translation quality is measured using **BLEU score** (0-100, higher is better):
- **60-100:** Excellent translation
- **40-60:** Good translation
- **20-40:** Fair translation
- **0-20:** Poor translation

## Evaluate Locally

Test your agent on the public dataset before submission:

```python
from translate import Env
from agent import Agent  # Your agent implementation

# Initialize environment and agent
env = Env(batch_size=10)
agent = Agent()
agent.reset()

# Run evaluation
total_score = 0
num_batches = 0

while not env.is_complete():
    task = env.get_next_task()
    if task is None:
        break

    # Agent predicts translations
    predictions = agent.predict(task['X_test'])

    # Evaluate against references
    score = env.evaluate(predictions, task['y_test'])
    total_score += score
    num_batches += 1

    print(f"Batch {num_batches}: BLEU = {score:.2f}")

# Final average score
avg_score = total_score / num_batches if num_batches > 0 else 0
print(f"\nAverage BLEU Score: {avg_score:.2f}")
```

## Versioning

Current version: **0.3**

**Changelog:**
- v0.3: Minor improvements and refinements
- v0.2: Replaced token overlap with BLEU metric (sacrebleu)
- v0.1: Initial release with token overlap metric
