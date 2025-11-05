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

try:
    from sacrebleu import corpus_bleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


class TranslateEnv:
    """
    Predict-only environment for English-German translation

    This environment is designed for testing pre-trained translation models.
    Unlike training environments, it only provides test data (English sentences)
    and evaluates the quality of German translations.
    """

    def __init__(
        self,
        batch_size: int = 10,
        dataset_path: Optional[str] = None,
        dataset: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the translation environment

        Args:
            batch_size: Number of translation pairs per batch
                       The dataset will be split into batches of this size
            dataset_path: Optional path to custom dataset JSON file
                         If provided, loads dataset from this path
            dataset: Optional pre-loaded dataset dictionary with 'pairs' key
                    If provided, uses this dataset directly

        Note: If both dataset_path and dataset are provided, dataset takes precedence.
              If neither is provided, uses default en_de_dataset.json
        """
        self.batch_size = batch_size
        self.current_batch = 0
        self.rng = np.random.RandomState()

        # Load translation dataset with priority: dataset > dataset_path > default
        if dataset is not None:
            # Use pre-loaded dataset
            self.dataset = dataset
        elif dataset_path is not None:
            # Load from custom path
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    self.dataset = json.load(f)
            except (FileNotFoundError, OSError) as e:
                raise RuntimeError(
                    f"Translation dataset not found at: {dataset_path}"
                ) from e
        else:
            # Load default dataset
            data_path = os.path.join(PKG_DIR, 'data')
            try:
                with open(os.path.join(data_path, 'en_de_dataset.json'), 'r', encoding='utf-8') as f:
                    self.dataset = json.load(f)
            except (FileNotFoundError, OSError) as e:
                raise RuntimeError(
                    "Default translation dataset not found. Please ensure the package is installed correctly."
                ) from e

        # Validate dataset structure
        if 'pairs' not in self.dataset:
            raise RuntimeError("Invalid dataset format: missing 'pairs' key")

        self.pairs = self.dataset['pairs']
        self.total_pairs = len(self.pairs)

        if self.total_pairs == 0:
            raise RuntimeError("Dataset is empty")

        # Calculate total number of batches
        self.num_batches = (self.total_pairs + self.batch_size - 1) // self.batch_size  # Ceiling division

        # Track current task data
        self.current_test_data = None
        self.current_references = None

    def set_seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility"""
        self.rng = np.random.RandomState(seed)

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """
        Get the next translation batch (predict-only mode)

        The dataset is split into batches of size batch_size.
        This method returns batches sequentially, ensuring all pairs are evaluated exactly once.

        Returns:
            Dictionary with:
                - X_test: Batch of English sentences to translate
                - y_test: Reference German translations (for evaluation only)
            Returns None when all batches are complete
        """
        if self.current_batch >= self.num_batches:
            return None

        # Calculate batch boundaries
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_pairs)

        # Extract English sentences and German references for this batch
        english_sentences = []
        german_references = []

        for idx in range(start_idx, end_idx):
            pair = self.pairs[idx]
            english_sentences.append(pair['en'])
            german_references.append(pair['de'])

        # Store references for evaluation
        self.current_references = german_references

        self.current_batch += 1

        # Return task in predict-only format (no X_train, y_train)
        return {
            'X_test': np.array(english_sentences, dtype=object),  # English sentences to translate
            'y_test': np.array(german_references, dtype=object)  # Reference translations (hidden from agent)
        }

    def evaluate(self, predictions: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Calculate translation quality using BLEU score

        Uses sacrebleu library for proper BLEU calculation with standard settings.
        BLEU score measures n-gram overlap between predictions and references.

        Args:
            predictions: Array of predicted German translations
            true_labels: Array of reference German translations

        Returns:
            BLEU score between 0 and 100 (higher is better)
        """
        if len(predictions) == 0:
            return 0.0

        if not BLEU_AVAILABLE:
            raise RuntimeError(
                "sacrebleu is required for BLEU evaluation but not installed. "
                "Install with: pip install sacrebleu"
            )

        # Convert numpy arrays to lists of strings
        pred_strings = [str(pred) for pred in predictions]
        ref_strings = [[str(ref)] for ref in true_labels]  # sacrebleu expects list of references per prediction

        # Calculate corpus BLEU score
        # sacrebleu returns score out of 100
        bleu = corpus_bleu(pred_strings, ref_strings)

        return float(bleu.score)

    def reset(self):
        """Reset environment to start from first batch"""
        self.current_batch = 0
        self.current_test_data = None
        self.current_references = None

    def is_complete(self) -> bool:
        """Check if all batches are complete"""
        return self.current_batch >= self.num_batches
