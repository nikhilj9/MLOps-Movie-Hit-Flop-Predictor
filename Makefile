.PHONY: train lint test format run

train:
	python src/train_model.py --config configs/train.yaml

lint:
	ruff check .

format:
	black .

test:
	pytest tests/

run:
	uvicorn src.main:app --reload
