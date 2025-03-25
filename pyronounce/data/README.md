# PyRonounce Data Directory

This directory contains data files used by PyRonounce.

## Files

- `default_model.pkl`: Pre-trained model for assessing word pronounceability. This file is included in the package distribution.

## Generated Files (not included in distribution)

The following files may be generated during runtime but are not included in the git repository or package distribution:

- `cmu_dict_cache.json`: A cached version of the CMU Pronouncing Dictionary, generated when NLTK is available.

## For Developers

If you're modifying the code:

1. Do not commit the cache files to the git repository.
2. Any additional data files that should be distributed with the package should be added to MANIFEST.in. 