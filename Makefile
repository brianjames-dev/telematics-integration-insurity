.PHONY: simulate ingest verify phase1 test clean data-clean

DRIVERS ?= 2
TRIPS   ?= 3
HZ      ?= 1.0
OUT     ?= data/pings

simulate:
	python -m src.simulator.generate_trips --drivers $(DRIVERS) --trips $(TRIPS) --hz $(HZ) --golden

ingest:
	python -m src.ingest.ingest --input "data/tmp/*.ndjson" --out "$(OUT)"

verify:
	python bin/verify_parquet.py --path $(OUT)

phase1: simulate ingest verify

test:
	pytest -q

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache

data-clean:
	rm -rf data/pings data/tmp
	mkdir -p data/tmp
