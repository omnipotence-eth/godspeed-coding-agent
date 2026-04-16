.PHONY: lint fix format type-check security test test-cov clean install all

lint:
	ruff check .
	ruff format --check .

fix:
	ruff check . --fix
	ruff format .

format:
	ruff format .

type-check:
	ty check src/ || mypy src/ --ignore-missing-imports

security:
	pip-audit
	bandit -r src/ -c pyproject.toml -ll || bandit -r src/ -ll

test:
	pytest --cov --cov-report=term-missing --cov-fail-under=80 -q

test-cov:
	pytest --cov --cov-report=term-missing --cov-report=html

clean:
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

install:
	uv tool install -e .

all: lint type-check security test
