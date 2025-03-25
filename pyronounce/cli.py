"""
Command-line interface for the pyronounce package.
"""

import sys
import json
import click
from .core import PronounceabilityAssessor

@click.command(help="Assess the pronounceability of English words.")
@click.argument("words", nargs=-1)
@click.option(
    "-d", "--detailed", 
    is_flag=True,
    help="Show detailed feature information."
)
@click.option(
    "-t", "--text",
    is_flag=True,
    help="Treat input as text rather than individual words."
)
@click.option(
    "-j", "--json",
    is_flag=True, 
    help="Output results as JSON."
)
@click.option(
    "-r", "--retrain",
    is_flag=True,
    help="Retrain the model before assessment."
)
def main(words, detailed, text, json, retrain):
    """
    Main entry point for the CLI.
    """
    # Get input
    if words:
        input_data = " ".join(words) if text else words
    else:
        # Read from stdin if no words provided
        input_data = sys.stdin.read().strip()
        if not text:
            input_data = input_data.split()
    
    # Initialize the assessor
    assessor = PronounceabilityAssessor()
    
    # Retrain if requested
    if retrain:
        from .model import train_and_save_default_model
        train_and_save_default_model()
        assessor = PronounceabilityAssessor()  # Reload with new model
    
    # Perform assessment
    if text:
        results = assessor.assess_text(input_data, detailed=detailed)
    else:
        if isinstance(input_data, tuple) or isinstance(input_data, list):
            results = [assessor.assess_word(word, detailed=detailed) for word in input_data]
        else:
            results = assessor.assess_word(input_data, detailed=detailed)
    
    # Output results
    if json:
        click.echo(json.dumps(results, indent=2))
    else:
        if text:
            click.echo(f"Text: '{results['text']}'")
            click.echo(f"Average score: {results['average_score']:.2f}")
            click.echo(f"Overall category: {results['overall_category']}")
            click.echo(f"Word count: {results['word_count']}")
            click.echo("\nWord-by-word analysis:")
            for word_result in results['words']:
                if word_result['score'] is not None:
                    click.echo(f"  '{word_result['word']}' ({word_result['ipa']}): {word_result['category']} (score: {word_result['score']:.2f})")
                else:
                    click.echo(f"  '{word_result['word']}': Error - {word_result['error']}")
        else:
            if isinstance(results, list):
                for result in results:
                    if result['score'] is not None:
                        click.echo(f"'{result['word']}' ({result['ipa']}): {result['category']} (score: {result['score']:.2f})")
                        if detailed and 'features' in result:
                            for name, value in result['features'].items():
                                click.echo(f"  {name}: {value:.2f}")
                    else:
                        click.echo(f"'{result['word']}': Error - {result['error']}")
            else:
                if results['score'] is not None:
                    click.echo(f"'{results['word']}' ({results['ipa']}): {results['category']} (score: {results['score']:.2f})")
                    if detailed and 'features' in results:
                        for name, value in results['features'].items():
                            click.echo(f"  {name}: {value:.2f}")
                else:
                    click.echo(f"'{results['word']}': Error - {results['error']}")

if __name__ == "__main__":
    main() 