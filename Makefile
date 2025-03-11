install:
	pip install -e .[dev]

test:
	pytest us_imputation_benchmarking/tests/ --maxfail=0
	pytest us_imputation_benchmarking/tests -s

format:
	black . -l 79
	linecheck . --fix
	isort us_imputation_benchmarking/