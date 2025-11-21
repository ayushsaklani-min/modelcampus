PYTHON ?= python
BENTOML ?= bentoml
BENTOML_HOME ?= $(HOME)/bentoml

NORMALIZE_SCRIPT := scripts/normalize_bento_install.py

.PHONY: build normalize-bento-install

build: ## Build the Bento and normalize generated install scripts.
	$(BENTOML) build
	$(MAKE) normalize-bento-install

normalize-bento-install: ## Convert CRLF install scripts under the local Bento store.
	$(PYTHON) $(NORMALIZE_SCRIPT) --bentoml-home "$(BENTOML_HOME)"

