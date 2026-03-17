PYTHON ?= python3
UV ?= uv

.PHONY: test-docs test-docs-local

test-docs:
	$(UV) run $(PYTHON) -m unittest tests.test_doctests

test-docs-local:
	PYTHONPATH=src $(PYTHON) -m unittest tests.test_doctests
