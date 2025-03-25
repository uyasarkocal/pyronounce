"""
Utility functions for the pyronounce package.
"""

import numpy as np

def word_to_ipa(word):
    """
    Convert a word to IPA notation using the pronouncing library.
    
    Args:
        word (str): The word to convert.
        
    Returns:
        tuple: (ipa_string, stress_markers)
    """
    # This is a simplified implementation - 
    # in a real project you would need to install the pronouncing library
    # import pronouncing
    
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
    word = word.lower()
    
    # Simplified implementation for demonstration
    # In a real project, you would use pronouncing.phones_for_word(word)
    arpabet = []
    for char in word:
        # Map common letters to approximate phonemes
        letter_to_phone = {
            'a': ['AE'], 'b': ['B'], 'c': ['K'], 'd': ['D'], 'e': ['EH'],
            'f': ['F'], 'g': ['G'], 'h': ['HH'], 'i': ['IH'], 'j': ['JH'],
            'k': ['K'], 'l': ['L'], 'm': ['M'], 'n': ['N'], 'o': ['AO'],
            'p': ['P'], 'q': ['K'], 'r': ['R'], 's': ['S'], 't': ['T'],
            'u': ['AH'], 'v': ['V'], 'w': ['W'], 'x': ['K', 'S'], 'y': ['Y'],
            'z': ['Z']
        }
        arpabet.extend(letter_to_phone.get(char.lower(), [char]))
            
    # Preserve stress information with separate stress marker
    ipa = ''
    stress_markers = []
    for i, p in enumerate(arpabet):
        # Extract stress marker
        if isinstance(p, str) and p[-1] in '012':
            stress = p[-1]
            phone = p[:-1]
            if stress == '1':  # Primary stress
                stress_markers.append(i)
        else:
            phone = p
            
        ipa += arpabet_to_ipa.get(phone, phone)
        
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
    vowels = set('æɛɪɑʌiueəoɔɝaɪeɪoʊaʊɔɪʊ')
    front_vowels = set('iɪeɛæ')
    back_vowels = set('uʊoɔɑ')
    diphthongs = set('aɪeɪoʊaʊɔɪ')
    
    stops = set('bdkɡpt')
    fricatives = set('fvszʃʒθð')
    affricates = set('tʃdʒ')
    nasals = set('mnŋ')
    liquids = set('lr')
    glides = set('wj')
    
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
    
    # Calculate unusual phoneme patterns
    unusual_phonemes = set('θðʃʒŋ')  # Sounds not common in most languages
    unusual_count = sum(1 for char in ipa if char in unusual_phonemes)
    
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
        
    return np.array([
        syllable_count / 4,  # Normalize to reduce impact
        max_cluster / 3,     # Normalize consonant clusters
        vowel_ratio,
        consonant_complexity,
        diphthong_count / 2,  # Normalize diphthongs
        stress_complexity / 2,
        length_complexity,
        unusual_ratio * 2     # Weight unusual sounds more heavily
    ]) 