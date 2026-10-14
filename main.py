"""Command-line entry point: python main.py --mode auto|hitl [--goal "..."]"""

from newsletter_agent.cli import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli())
