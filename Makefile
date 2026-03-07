.PHONY: serve test lint docker-build ingest export-onnx

serve:
	uvicorn api.app:app --reload

test:
	python -m pytest tests/ -v

lint:
	ruff check .

docker-build:
	docker build -t medical-ai .

ingest:
	python rag/ingest.py

export-onnx:
	python scripts/export_onnx.py --model best.pt
