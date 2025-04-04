install:
	pip install -e ".[dev,docs]"

test:
	pytest us_imputation_benchmarking/tests/ --cov=us_imputation_benchmarking --cov-report=xml --maxfail=0

format:
	black . -l 79
	linecheck . --fix
	isort us_imputation_benchmarking/

documentation:
	cd docs && jupyter-book build .

build:
	pip install build
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info/
	rm -rf docs/_build/
