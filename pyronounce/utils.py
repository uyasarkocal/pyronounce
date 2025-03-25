"""
Utility functions for the pyronounce package.
"""

import numpy as np
import re
import warnings
import os
import json
from importlib.util import find_spec

# Check if nltk is installed
NLTK_AVAILABLE = find_spec("nltk") is not None

# Path to store our CMU dict cache
CMU_DICT_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'cmu_dict_cache.json')

# Load or create CMU Dictionary cache
def load_cmu_dict():
    """Load the CMU Pronouncing Dictionary or a cached version of it."""
    if os.path.exists(CMU_DICT_CACHE_PATH):
        try:
            with open(CMU_DICT_CACHE_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # If cache is corrupted, reload from NLTK
    
    cmu_dict = {}
    
    if NLTK_AVAILABLE:
        try:
            import nltk
            from nltk.corpus import cmudict
            
            # Ensure CMU dict is downloaded
            try:
                nltk.data.find('corpora/cmudict')
            except LookupError:
                nltk.download('cmudict', quiet=True)
            
            # Load the dictionary
            raw_dict = cmudict.dict()
            
            # Process into a more usable format
            for word, pronunciations in raw_dict.items():
                # Use the first pronunciation for simplicity
                if pronunciations:
                    cmu_dict[word] = pronunciations[0]
            
            # Cache the results
            os.makedirs(os.path.dirname(CMU_DICT_CACHE_PATH), exist_ok=True)
            with open(CMU_DICT_CACHE_PATH, 'w') as f:
                json.dump(cmu_dict, f)
                
            return cmu_dict
        except Exception as e:
            warnings.warn(f"Failed to load CMU dictionary: {e}")
    
    # Return empty dict if loading failed
    return cmu_dict

# Load CMU Dictionary
CMU_DICT = load_cmu_dict()

def word_to_ipa(word):
    """
    Convert a word to IPA notation using CMU Dictionary.
    
    Args:
        word (str): The word to convert.
        
    Returns:
        tuple: (ipa_string, stress_markers)
    """
    word = word.lower().strip()
    
    # ARPABET to IPA mapping
    arpabet_to_ipa = {
        'AA': 'ɑ', 'AE': 'æ', 'AH': 'ə', 'AO': 'ɔ', 'AW': 'aʊ',
        'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
        'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ',
        'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
        'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
        'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ',
        'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
        'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
    }
    
    # Special cases dictionary for words CMU might not handle well
    special_cases = {
        "xylophone": ("Z AY L AH F OW N", [0]),
        "tschüss": ("CH UW S", [0]),
        "zeitgeist": ("T S AY T G AY S T", [0, 3]),
        "phantasy": ("F AE N T AH S IY", [0]),
    }
    
    # Get phonemes from CMU dict or special cases
    phonemes = None
    stress_positions = []
    
    # Check special cases first
    if word in special_cases:
        phoneme_str, stress_positions = special_cases[word]
        phonemes = phoneme_str.split()
    # Then check CMU dictionary
    elif word in CMU_DICT:
        phonemes = CMU_DICT[word]
    
    # If we have phonemes, convert to IPA
    if phonemes:
        ipa = ""
        stress_markers = []
        current_pos = 0
        
        for phoneme in phonemes:
            # Extract stress markers (numbers at the end of vowel phonemes)
            if any(phoneme.startswith(v) for v in ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']):
                if phoneme[-1].isdigit():
                    stress_level = int(phoneme[-1])
                    phoneme = phoneme[:-1]
                    if stress_level == 1:  # Primary stress
                        stress_markers.append(current_pos)
            
            # Convert to IPA
            if phoneme in arpabet_to_ipa:
                ipa_char = arpabet_to_ipa[phoneme]
                ipa += ipa_char
                current_pos += len(ipa_char)
            else:
                # For unknown phonemes
                ipa += phoneme.lower()
                current_pos += len(phoneme)
        
        # Use predefined stress positions for special cases if available
        if stress_positions and word in special_cases:
            stress_markers = stress_positions
            
        return f"/{ipa}/", stress_markers
    
    # Fallback implementation if CMU dict lookup fails
    # Simple character-to-sound mapping with some basic rules
    char_to_sound = {
        'a': 'æ', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'ɛ',
        'f': 'f', 'g': 'ɡ', 'h': 'h', 'i': 'ɪ', 'j': 'dʒ',
        'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'ɑ',
        'p': 'p', 'q': 'k', 'r': 'r', 's': 's', 't': 't',
        'u': 'ʌ', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'j',
        'z': 'z'
    }
    
    # Apply some basic rules
    ipa = ""
    i = 0
    while i < len(word):
        # Handle digraphs and special cases
        if i < len(word) - 1:
            digraph = word[i:i+2]
            if digraph == 'ph':
                ipa += 'f'
                i += 2
                continue
            elif digraph == 'th':
                ipa += 'θ'
                i += 2
                continue
            elif digraph == 'sh':
                ipa += 'ʃ'
                i += 2
                continue
            elif digraph == 'ch':
                ipa += 'tʃ'
                i += 2
                continue
            elif digraph == 'qu':
                ipa += 'kw'
                i += 2
                continue
        
        # Handle 'c' based on context
        if word[i] == 'c' and i < len(word) - 1 and word[i+1] in 'eiy':
            ipa += 's'
        elif word[i] in char_to_sound:
            ipa += char_to_sound[word[i]]
        else:
            ipa += word[i]
        
        i += 1
    
    # Approximate stress marker on first syllable
    stress_markers = []
    vowels = set('æɛɪɑʌiueəoɔɝaɪeɪoʊaʊɔɪʊ')
    for i, char in enumerate(ipa):
        if char in vowels:
            stress_markers.append(i)
            break
    
    return f"/{ipa}/", stress_markers

def extract_features(ipa, stress_markers):
    """
    Extract phonetic features from an IPA representation.
    
    Args:
        ipa (str): IPA representation of the word.
        stress_markers (list): Positions of primary stress.
        
    Returns:
        np.ndarray: Feature vector.
    """
    ipa = ipa.strip('/')
    
    # Enhanced phonetic categories
    vowels = set('æɛɪɑʌiueəoɔɝaɪeɪoʊaʊɔɪʊyøœ')
    front_vowels = set('iɪeɛæy')
    back_vowels = set('uʊoɔɑø')
    diphthongs = set('aɪeɪoʊaʊɔɪɔʏ')
    
    stops = set('bdkɡpt')
    fricatives = set('fvszʃʒθðçxɣχ')
    affricates = set('tʃdʒpftsdzψ')
    nasals = set('mnŋɱɲ')
    liquids = set('lrɾɹʁ')
    glides = set('wjɥ')
    
    # Count syllables more accurately
    syllable_count = 0
    prev_was_vowel = False
    for char in ipa:
        if char in vowels:
            if not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = True
        else:
            prev_was_vowel = False
            
    # Calculate maximum consonant cluster
    consonants = stops.union(fricatives, affricates, nasals, liquids, glides)
    max_cluster = 0
    current_cluster = 0
    
    for char in ipa:
        if char in consonants:
            current_cluster += 1
            max_cluster = max(max_cluster, current_cluster)
        else:
            current_cluster = 0
    
    # Calculate unusual phoneme patterns - expanded list
    unusual_phonemes = set('θðʃʒŋçxɣχɱɲɾɹʁψɥy')  # Sounds not common in many languages
    unusual_count = sum(1 for char in ipa if char in unusual_phonemes)
    
    # Find consonant clusters
    consonant_clusters = re.findall(r'[bdkɡptfvszʃʒθðçxɣχtʃdʒpftsdzmnŋɱɲlrɾɹʁwj]{3,}', ipa)
    complex_cluster_factor = sum(len(cluster) for cluster in consonant_clusters) / 3 if consonant_clusters else 0
    
    # Count phonetic elements
    vowel_count = sum(1 for char in ipa if char in vowels)
    diphthong_count = sum(1 for i in range(len(ipa)-1) if ipa[i:i+2] in diphthongs)
    fricative_count = sum(1 for char in ipa if char in fricatives)
    affricate_count = sum(1 for i in range(len(ipa)-1) if ipa[i:i+2] in affricates)
    
    # Calculate ratios
    total = len(ipa)
    vowel_ratio = vowel_count / total if total > 0 else 0
    consonant_complexity = (fricative_count + 2*affricate_count) / total if total > 0 else 0
    unusual_ratio = unusual_count / total if total > 0 else 0
    
    # Calculate stress pattern complexity
    stress_complexity = len(stress_markers)
    
    # Assess word length complexity - with less penalty for medium words
    length_complexity = min(1.0, syllable_count / 5)
    
    # Calculate complexity from phonotactic rarity
    phonotactic_complexity = max_cluster / 3 + complex_cluster_factor
        
    return np.array([
        syllable_count / 4,             # Normalize to reduce impact
        phonotactic_complexity,         # Enhanced consonant cluster analysis
        vowel_ratio,
        consonant_complexity,
        diphthong_count / 2,            # Normalize diphthongs
        stress_complexity / 2,
        length_complexity,
        unusual_ratio * 2.5             # Weight unusual sounds more heavily
    ]) 