.PHONY: install generate train evaluate test lint api

install:
	python -m pip install -e ".[dev]"

generate:
	python scripts/generate_data.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

test:
	python -m pytest

lint:
	ruff check src app scripts tests

api:
	uvicorn app.main:app --reload
