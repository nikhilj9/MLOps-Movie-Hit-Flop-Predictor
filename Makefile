.PHONY: install lint test run

install:
	pip install -r requirements.txt

lint:
	ruff check . --fix

test:
	pytest

run:
	uvicorn src.main:app --reload