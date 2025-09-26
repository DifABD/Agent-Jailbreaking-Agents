.PHONY: help install install-dev test lint format clean run-api run-dashboard db-upgrade db-downgrade

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=src --cov-report=html --cov-report=term

lint:  ## Run linting
	flake8 src tests
	mypy src

format:  ## Format code
	black src tests
	isort src tests

format-check:  ## Check code formatting
	black --check src tests
	isort --check-only src tests

clean:  ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/

run-api:  ## Run the FastAPI server
	uvicorn src.api.main:app --reload

run-dashboard:  ## Run the Streamlit dashboard
	streamlit run src/dashboard/app.py

db-upgrade:  ## Run database migrations
	alembic upgrade head

db-downgrade:  ## Rollback database migrations
	alembic downgrade -1

db-revision:  ## Create a new database migration
	alembic revision --autogenerate -m "$(MESSAGE)"

setup-dev:  ## Set up development environment
	python -m venv venv
	@echo "Activate virtual environment with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
	@echo "Then run: make install-dev"

pre-commit-install:  ## Install pre-commit hooks
	pre-commit install

pre-commit-run:  ## Run pre-commit hooks on all files
	pre-commit run --all-files