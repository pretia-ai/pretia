.PHONY: check lint format typecheck test build clean

check: lint format typecheck test
	@echo "All checks passed."

lint:
	uv run ruff check pretia/ tests/

format:
	uv run ruff format --check pretia/ tests/

typecheck:
	uv run pyright pretia/

test:
	uv run pytest tests/unit/ -v --tb=short

build: check
	rm -rf dist/ build/
	uv run python -m build

clean:
	rm -rf dist/ build/ *.egg-info
