"""Command line interface for running experiments."""

import click
from src.config import get_settings


@click.command()
@click.option('--persuader', default=None, help='Persuader model name')
@click.option('--persuadee', default=None, help='Persuadee model name')
@click.option('--claim', required=True, help='Claim to debate')
@click.option('--max-turns', default=None, type=int, help='Maximum conversation turns')
def main(persuader: str, persuadee: str, claim: str, max_turns: int):
    """Run a single experiment with the specified parameters."""
    settings = get_settings()
    
    # Use defaults from settings if not provided
    persuader = persuader or settings.default_persuader_model
    persuadee = persuadee or settings.default_persuadee_model
    max_turns = max_turns or settings.max_turns
    
    click.echo(f"Starting experiment:")
    click.echo(f"  Persuader: {persuader}")
    click.echo(f"  Persuadee: {persuadee}")
    click.echo(f"  Claim: {claim}")
    click.echo(f"  Max turns: {max_turns}")
    
    # TODO: Implement actual experiment execution
    click.echo("Experiment execution not yet implemented.")


if __name__ == "__main__":
    main()