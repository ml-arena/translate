"""
English-German Translation Environment for Predict-Only Supervised Learning

This environment provides a predict-only workflow where agents receive English
sentences and must predict their German translations. No training data is provided
during the competition - agents must use pre-trained models.
"""
import numpy as np
import os
import json
from typing import Optional, Dict, Any
from translate import PKG_DIR


class TranslateEnv:
    """
    Predict-only environment for English-German translation

    This environment is designed for testing pre-trained translation models.
    Unlike training environments, it only provides test data (English sentences)
    and evaluates the quality of German translations.
    """

    def __init__(self, number_episodes: int = 20):
        """
        Initialize the translation environment

        Args:
            number_episodes: Number of translation tasks to evaluate
        """
        self.number_episodes = number_episodes
        self.current_episode = 0
        self.rng = np.random.RandomState()

        # Load translation dataset
        data_path = os.path.join(PKG_DIR, 'data')
        try:
            with open(os.path.join(data_path, 'en_de_dataset.json'), 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError(
                "Translation dataset not found. Please run the data preparation script first:\n"
                "python tools/prepare_data.py"
            ) from e

        # Validate dataset structure
        if 'pairs' not in self.dataset:
            raise RuntimeError("Invalid dataset format: missing 'pairs' key")

        self.pairs = self.dataset['pairs']
        self.total_pairs = len(self.pairs)

        if self.total_pairs == 0:
            raise RuntimeError("Dataset is empty")

        # Calculate samples per task
        self.samples_per_task = max(10, self.total_pairs // number_episodes)

        # Track current task data
        self.current_test_data = None
        self.current_references = None

    def set_seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility"""
        self.rng = np.random.RandomState(seed)

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the next translation task (predict-only mode)

        Returns:
            Dictionary with:
                - X_test: List of English sentences to translate
                - y_test: Reference German translations (for evaluation only)
            Returns None when all episodes are complete
        """
        if self.current_episode >= self.number_episodes:
            return None

        # Sample random sentences for this task
        indices = self.rng.choice(
            self.total_pairs,
            size=min(self.samples_per_task, self.total_pairs),
            replace=False
        )

        # Extract English sentences and German references
        english_sentences = []
        german_references = []

        for idx in indices:
            pair = self.pairs[int(idx)]
            english_sentences.append(pair['en'])
            german_references.append(pair['de'])

        # Store references for evaluation
        self.current_references = german_references

        self.current_episode += 1

        # Return task in predict-only format (no X_train, y_train)
        return {
            'X_test': np.array(english_sentences, dtype=object),  # English sentences to translate
            'y_test': np.array(german_references, dtype=object)  # Reference translations (hidden from agent)
        }

    def evaluate(self, predictions: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Calculate translation quality using BLEU-like token overlap metric

        This is a simplified evaluation metric based on word overlap.
        In production, you would use proper BLEU, chrF, or other MT metrics.

        Args:
            predictions: Array of predicted German translations
            true_labels: Array of reference German translations

        Returns:
            Score between 0 and 1 (higher is better)
        """
        if len(predictions) == 0:
            return 0.0

        scores = []
        for pred, ref in zip(predictions, true_labels):
            # Convert to lowercase and tokenize by whitespace
            pred_tokens = set(str(pred).lower().split())
            ref_tokens = set(str(ref).lower().split())

            if len(ref_tokens) == 0:
                scores.append(0.0)
                continue

            # Calculate token overlap (simplified BLEU-like metric)
            overlap = len(pred_tokens & ref_tokens)
            precision = overlap / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = overlap / len(ref_tokens)

            # F1 score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            scores.append(f1)

        return float(np.mean(scores))

    def reset(self):
        """Reset environment for new set of episodes"""
        self.current_episode = 0
        self.current_test_data = None
        self.current_references = None

    def is_complete(self) -> bool:
        """Check if all episodes are complete"""
        return self.current_episode >= self.number_episodes
