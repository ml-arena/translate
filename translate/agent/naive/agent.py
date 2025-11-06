"""
Naive Vocabulary-Based Agent for English-German Translation
Makes simple word-by-word translations using a basic dictionary - useful as a baseline
"""
import numpy as np
import re


class Agent:
    """Naive agent that uses a simple vocabulary dictionary for word-by-word translation"""

    def __init__(self, seed: int = None):
        """
        Initialize the naive vocabulary-based agent

        Args:
            seed: Random seed (not used by this agent but kept for API consistency)
        """
        # Basic English-German vocabulary dictionary
        self.vocab = {
            # Greetings
            'hello': 'hallo',
            'hi': 'hallo',
            'good': 'guten',
            'morning': 'morgen',
            'evening': 'abend',
            'afternoon': 'nachmittag',
            'night': 'nacht',
            'goodbye': 'auf wiedersehen',
            'bye': 'tschüss',

            # Courtesy
            'please': 'bitte',
            'thank': 'danke',
            'thanks': 'danke',
            'you': 'du',
            'welcome': 'willkommen',
            'sorry': 'tut mir leid',
            'excuse': 'entschuldigung',

            # Questions
            'how': 'wie',
            'what': 'was',
            'where': 'wo',
            'when': 'wann',
            'who': 'wer',
            'why': 'warum',
            'which': 'welche',

            # Common verbs
            'are': 'bist',
            'is': 'ist',
            'am': 'bin',
            'do': 'tun',
            'have': 'haben',
            'go': 'gehen',
            'come': 'kommen',
            'see': 'sehen',
            'want': 'wollen',
            'need': 'brauchen',
            'like': 'mögen',
            'love': 'lieben',
            'speak': 'sprechen',
            'understand': 'verstehen',
            'know': 'wissen',
            'think': 'denken',

            # Pronouns
            'i': 'ich',
            'me': 'mich',
            'my': 'mein',
            'we': 'wir',
            'us': 'uns',
            'our': 'unser',
            'your': 'dein',
            'he': 'er',
            'she': 'sie',
            'it': 'es',
            'they': 'sie',
            'them': 'ihnen',

            # Common nouns
            'day': 'tag',
            'time': 'zeit',
            'name': 'name',
            'water': 'wasser',
            'food': 'essen',
            'coffee': 'kaffee',
            'tea': 'tee',
            'beer': 'bier',
            'wine': 'wein',
            'bread': 'brot',
            'house': 'haus',
            'car': 'auto',
            'train': 'zug',
            'bus': 'bus',
            'street': 'straße',
            'city': 'stadt',
            'country': 'land',
            'year': 'jahr',
            'month': 'monat',
            'week': 'woche',

            # Numbers
            'one': 'eins',
            'two': 'zwei',
            'three': 'drei',
            'four': 'vier',
            'five': 'fünf',
            'six': 'sechs',
            'seven': 'sieben',
            'eight': 'acht',
            'nine': 'neun',
            'ten': 'zehn',

            # Adjectives
            'nice': 'schön',
            'beautiful': 'schön',
            'good': 'gut',
            'bad': 'schlecht',
            'big': 'groß',
            'small': 'klein',
            'new': 'neu',
            'old': 'alt',
            'happy': 'glücklich',
            'sad': 'traurig',
            'hot': 'heiß',
            'cold': 'kalt',
            'easy': 'einfach',
            'difficult': 'schwierig',

            # Prepositions
            'in': 'in',
            'on': 'auf',
            'at': 'bei',
            'to': 'zu',
            'from': 'von',
            'with': 'mit',
            'without': 'ohne',
            'for': 'für',
            'about': 'über',
            'before': 'vor',
            'after': 'nach',

            # Other
            'yes': 'ja',
            'no': 'nein',
            'not': 'nicht',
            'very': 'sehr',
            'much': 'viel',
            'many': 'viele',
            'all': 'alle',
            'some': 'einige',
            'here': 'hier',
            'there': 'dort',
            'now': 'jetzt',
            'today': 'heute',
            'tomorrow': 'morgen',
            'yesterday': 'gestern',
        }

    def reset(self):
        """Reset the agent for a new task/simulation"""
        # Naive agent has no state to reset
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the agent on the provided data
        Naive agent doesn't train, just ignores the data

        Args:
            X_train: Training English sentences
            y_train: Training German translations
        """
        # Naive agent doesn't learn from data
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Translate English sentences to German using word-by-word vocabulary lookup

        Args:
            X_test: Array of English sentences to translate

        Returns:
            Array of German translations (naive word-by-word translation)
        """
        translations = []

        for sentence in X_test:
            # Convert to lowercase and split into words
            sentence_str = str(sentence).lower()

            # Remove punctuation but keep it for later
            # Match words and punctuation separately
            tokens = re.findall(r'\w+|[^\w\s]', sentence_str)

            translated_words = []
            for token in tokens:
                if re.match(r'\w+', token):
                    # It's a word - try to translate it
                    translated_word = self.vocab.get(token, token)
                    translated_words.append(translated_word)
                else:
                    # It's punctuation - keep it as is
                    translated_words.append(token)

            # Join the translated words
            # Add space before punctuation for readability
            translation = ' '.join(translated_words)

            # Clean up spacing around punctuation
            translation = re.sub(r'\s+([.,!?])', r'\1', translation)

            # Capitalize first letter
            if translation:
                translation = translation[0].upper() + translation[1:]

            translations.append(translation)

        return np.array(translations, dtype=object)
