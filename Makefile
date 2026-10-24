CANDIDATE ?= candidate_001_fused_cuda
KDA := kda-attention-sm89

.PHONY: help install build validate test bench plot lint
help:
	@echo "targets: install build validate test bench plot lint   (CANDIDATE=$(CANDIDATE))"

install:
	pip install -r requirements-dev.txt

build:
	bash $(KDA)/candidates/$(CANDIDATE)/compile.sh

validate:
	cd $(KDA) && python3 validate.py $(CANDIDATE)

test:
	python3 -m pytest

bench:
	cd $(KDA) && python3 benchmark.py $(CANDIDATE)

plot:
	python3 scripts/plot_speedup.py

lint:
	ruff check .
