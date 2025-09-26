# Agent Jailbreak Research System

A systematic investigation framework for LLM agent jailbreaking through multi-turn persuasive dialogue.

## Overview

This project implements a dual-agent conversational framework using LangGraph to simulate and measure persuasive attacks between LLM agents in a controlled environment. The system analyzes the effectiveness of various unethical persuasion strategies and the resistance capabilities of different models.

## Features

- **Multi-Agent Framework**: Dual-agent system with Persuader (attacker) and Persuadee (victim) agents
- **LangGraph Workflows**: Stateful conversation management with conditional routing
- **Model Support**: GPT-4o, Llama-3.3-70B, and Llama-Guard-2-8B integrations
- **Safety Evaluation**: Automated safety classification with human validation
- **Strategy Analysis**: Systematic categorization of persuasion strategies
- **Comprehensive Metrics**: Normalized Change calculation and jailbreak success determination
- **Data Management**: SQLAlchemy ORM with Alembic migrations
- **API Service**: FastAPI endpoints for experiment management
- **Visualization**: Interactive dashboards with Streamlit and Plotly

## Requirements

- Python 3.10 or higher
- PostgreSQL (production) or SQLite (development)
- OpenAI API access for GPT-4o
- Hugging Face access for Llama models

## Installation

### Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd agent-jailbreak-research
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Initialize the database:
```bash
alembic upgrade head
```

### Production Setup

1. Install production dependencies:
```bash
pip install -e ".[production]"
```

2. Configure PostgreSQL database and update environment variables

3. Run database migrations:
```bash
alembic upgrade head
```

## Usage

### Running Experiments

Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload
```

Launch the Streamlit dashboard:
```bash
streamlit run src/dashboard/app.py
```

### Command Line Interface

Run a single experiment:
```bash
ajr-experiment --persuader gpt-4o --persuadee llama-3.3-70b --claim "Your claim here"
```

### API Endpoints

- `POST /experiments/` - Create new experiment
- `GET /experiments/{id}` - Get experiment results
- `GET /experiments/` - List experiments with filtering
- `POST /experiments/batch` - Run batch experiments

## Development

### Code Quality

Run tests:
```bash
pytest
```

Format code:
```bash
black src tests
isort src tests
```

Type checking:
```bash
mypy src
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pre-commit install
```

## Architecture

The system is built on modern Python technologies:

- **LangGraph**: Stateful workflow orchestration
- **LangChain**: LLM integrations and prompt management
- **FastAPI**: High-performance API framework
- **SQLAlchemy**: Database ORM with async support
- **Pydantic**: Data validation and serialization
- **Streamlit**: Interactive dashboard
- **Plotly**: Data visualization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{agent_jailbreak_research,
  title={Agent Jailbreak Research System},
  author={Research Team},
  year={2024},
  url={https://github.com/example/agent-jailbreak-research}
}
```