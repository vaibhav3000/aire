# Developer entry points. Works with GNU make from Git Bash on Windows and from
# a Linux shell; every Python command runs in the locked uv environment.
#
#   make setup      create .venv from uv.lock with every extra
#   make check      the full gate CI runs (lint, types, tests, demo drift, docs)
#
# Run "make help" for the rest.

UV ?= uv
# --all-extras on every run as well: a plain "uv run" re-syncs to the default
# extras and would uninstall the dev tools and provider SDKs.
RUN := $(UV) run --locked --all-extras
REPORT := AIRE_AI_Reliability_Evaluation_Engine
IMAGE ?= aire

.PHONY: help setup lint format typecheck test validate demo docs docs-check gates \
	calibration drift check figures report docker

help:
	@echo "setup       install the locked environment (.venv) with every extra"
	@echo "lint        ruff check and ruff format --check"
	@echo "format      apply ruff format"
	@echo "typecheck   pyright"
	@echo "test        pytest"
	@echo "demo        regenerate results/, reports/*.html and demo-site/data.js"
	@echo "docs        regenerate the numbers in README, FAQ and report"
	@echo "check       full gate: lint, types, tests, demo drift, docs, gates"
	@echo "figures     rebuild the report figures (PDF) and their README images (PNG)"
	@echo "report      build the figures, then the technical report PDF with latexmk"
	@echo "docker      build the gate image and gate v2 against v1 in it"

setup:
	$(UV) sync --locked --all-extras

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff format .

typecheck:
	$(RUN) pyright

test:
	$(RUN) python -m pytest -q

validate:
	$(RUN) python demo/validate_eval_set.py

demo:
	$(RUN) python demo/run_demo.py
	$(RUN) python demo/judge_calibration/calibrate_judges.py
	$(RUN) python demo-site/build.py

docs:
	$(RUN) python scripts/sync_docs.py
	$(RUN) python scripts/figure_data.py

docs-check:
	$(RUN) python scripts/sync_docs.py --check
	$(RUN) python scripts/figure_data.py --check

# v2 must pass against v1 and the planted v3 regression must be blocked (exit 1).
gates:
	$(RUN) aire-gate results/v1_baseline results/v2_improved --ignore latency_ms
	$(RUN) aire-gate results/v1_baseline results/v3_fluent_regression --ignore latency_ms \
		> /dev/null; code=$$?; \
		if [ "$$code" -ne 1 ]; then echo "expected the v3 gate to exit 1, got $$code"; exit 1; fi; \
		echo "v3 blocked as expected"

# Regenerating the committed artifacts must not change them.
drift: demo
	git diff --exit-code -- results reports demo-site/data.js

check: lint typecheck test validate drift docs-check gates
	@echo "full gate passed"

# Each figure has one standalone TikZ/pgfplots source in reports/figures. Its PDF
# goes into the report; its PNG (pdftoppm, opaque white) goes into the README.
FIGURES := architecture gate_flow paired_deltas runner

figures:
	$(RUN) python scripts/figure_data.py
	cd reports/figures && for f in $(FIGURES); do \
		pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build $$f.tex \
			> /dev/null || { tail -n 30 build/$$f.log; exit 1; }; \
		cp build/$$f.pdf $$f.pdf; \
		pdftoppm -r 300 -png -singlefile $$f.pdf $$f; \
	done

report: figures
	cd reports && latexmk -pdf -interaction=nonstopmode -halt-on-error $(REPORT).tex

docker:
	docker info > /dev/null
	docker build -t $(IMAGE) .
	docker run --rm -v "$(CURDIR)/results:/data:ro" $(IMAGE) \
		/data/v1_baseline /data/v2_improved --ignore latency_ms
