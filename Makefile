.PHONY: check-all
check-all: check-format check-mypy check-pylint check-tests check-pydocstyle docs

.PHONY: check-format
check-format:
	isort --check oqpy
	black --check --diff oqpy

.PHONY: format
format:
	isort oqpy
	black oqpy

.PHONY: check-mypy
check-mypy:
	mypy oqpy

.PHONY: check-pylint
check-pylint:
	pylint oqpy

.PHONY: check-pydocstyle
check-pydocstyle:
	pydocstyle oqpy

.PHONY: docs
docs:
	make -C docs clean html

.PHONY: docs-clean
docs-clean:
	make -C docs clean

.PHONY: open-docs
open-docs:
	open docs/_build/html/index.html

.PHONY: check-tests
check-tests:
	pytest --cov=oqpy -vv --color=yes tests

.PHONY: check-citation
check-citation:
	cffconvert --validate

.PHONY: install-poetry
install-poetry:
	command -v curl >/dev/null 2>&1 || { echo >&2 "please install curl and retry."; exit 1; }
	curl -sSL https://install.python-poetry.org | python - $(poetry_version)
