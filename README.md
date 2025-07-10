# LLM Reasoning

## Setup

1. (Recommended) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

This will install [dspy](https://pypi.org/project/dspy/) and [openai](https://pypi.org/project/openai/).

## Project Structure

- `llm_reasoning/` - Main package for generalized LLM-guided search
  - `core.py` - Generic state, action, and solver abstractions
  - `tasks/` - Task-specific implementations (e.g., Tower of Hanoi, 8-Puzzle)
  - `models.py` - Shared Pydantic models for structured LLM input/output
  - `signatures.py` - Generic and task-specific DSPy signatures
  - `__main__.py` - CLI entry point for running solvers

- `tower_of_hanoi.py` - (Legacy) Standalone Tower of Hanoi script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Adding a New Task

1. Create a new file in `llm_reasoning/tasks/` (e.g., `eight_puzzle.py`).
2. Define your state, action, and evaluation models in `models.py` if needed.
3. Implement task-specific logic by subclassing the generic abstractions in `core.py`.
4. Add or adapt DSPy signatures in `signatures.py` for your task.
5. Register your task in `__main__.py` for CLI access.

See the Tower of Hanoi implementation in `llm_reasoning/tasks/tower_of_hanoi.py` for an example. 