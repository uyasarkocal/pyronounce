# PyRonounce

A Python package to assess the pronounceability of English words and text.

## Overview

PyRonounce analyzes words to determine how difficult they are to pronounce based on various phonetic features. It can be used to:

- Evaluate how easy or difficult a word is to pronounce
- Analyze text to determine its overall pronounceability
- Extract detailed phonetic feature information

The package uses a machine learning model trained on English words of varying pronounceability difficulty.

## Installation

You can install PyRonounce using pip:

```bash
pip install pyronounce
```

Or using UV for improved performance:

```bash
uv pip install pyronounce
```

To install from source:

```bash
git clone https://github.com/yourusername/pyronounce.git
cd pyronounce
pip install -e .
```

## Usage

### Command Line

PyRonounce can be used from the command line to assess words or text:

```bash
# Assess individual words
pyronounce hello world

# Assess text
pyronounce -t "This is some text to analyze"

# Show detailed feature information
pyronounce -d complicated

# Output JSON
pyronounce -j antidisestablishmentarianism

# Read from stdin
echo "supercalifragilisticexpialidocious" | pyronounce
```

### Python API

```python
import pyronounce

# Assess a single word
result = pyronounce.assess_word("complicated")
print(f"Score: {result['score']}, Category: {result['category']}")

# Assess with detailed feature information
detailed = pyronounce.assess_word("complicated", detailed=True)
features = detailed['features']

# Assess a text
text_result = pyronounce.assess_text("This is a sample text to analyze")
print(f"Average score: {text_result['average_score']}")
print(f"Overall category: {text_result['overall_category']}")
```

### Advanced Usage

You can create your own instance of the `PronounceabilityAssessor` class:

```python
from pyronounce import PronounceabilityAssessor

# Create with default model
assessor = PronounceabilityAssessor()

# Create with custom model
custom_assessor = PronounceabilityAssessor(model_path="/path/to/model.pkl")

# Get feature importance
importance = assessor.get_feature_importance()
```

## Features

PyRonounce evaluates words based on these phonetic features:

- **Syllable count**: Number of syllables
- **Consonant clusters**: Sequences of consonants without vowels
- **Vowel ratio**: Proportion of vowels to consonants
- **Consonant complexity**: Presence of complex consonants like fricatives
- **Diphthongs**: Two vowel sounds in one syllable
- **Stress patterns**: Placement of stress in the word
- **Word length**: Overall length consideration
- **Unusual sounds**: Phonemes not common in many languages

## Difficulty Categories

Words are classified into these categories:

- **Very Easy** (score > 0.85)
- **Easy** (score > 0.65)
- **Moderate** (score > 0.45)
- **Hard** (score > 0.25)
- **Very Hard** (score <= 0.25)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
