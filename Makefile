.PHONY: lint format type-check security test test-cov clean

lint:
	ruff check . --fix
	ruff format .

format:
	ruff format .

type-check:
	ty check src/ || mypy src/

security:
	pip-audit
	bandit -r src/ -c pyproject.toml || bandit -r src/

test:
	pytest -x -q

test-cov:
	pytest --cov --cov-report=term-missing --cov-report=html

clean:
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: lint type-check security test
