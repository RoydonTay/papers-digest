.PHONY: install test dry-run run lint

install:
	pip install -r requirements.txt

test:
	pytest

dry-run:
	python -m src.main --dry-run

run:
	python -m src.main
